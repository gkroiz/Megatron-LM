# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union
from datetime import datetime

import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model, get_model_type

import nvtx

# Types
Shape = Union[List[int], torch.Size]

def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of
        modules in the case of interleaved pipeline parallelism.

    num_microbatches (int, required):
        The number of microbatches to go through

    dtype (required when using pipeline parallelism): dtype used in
        p2p communication, usually params_dtype

    tensor_shape (required when using pipeline parallelism): Shape of
        tensor. The tensor is expected to be 3D and its order of
        dimension is supposed to be ``(sequence, batch, hidden)``.

    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    grad_scaler (optional, default=None): If using loss scaling,
        this function should take the loss and return the scaled
        loss. If None, no function is called on the loss.

    sequence_parallel (optional, default=False):
        Set to :obj:`True` for this function to handle sequence
        length.  When :obj:`True`, the sequence length on each tensor
        model parallel rank is updated to
        :math:`original\_sequence\_length /
        tensor\_model\_parallel\_world\_size`.
        TODO: Do we need this? Just roll into tensor_shape arg?

    overlap_p2p_comm (optional, default=False): When True
        some of the peer to peer communication for pipeline
        parallelism will overlap with computation. Must be False if
        batch_p2p_comm is true.

    batch_p2p_comm (optional, default=True): When true use
        batch_isend_irecv, otherwise use individual isend and irecv
        calls. Must be false if overlap_p2p_comm is True.

    forward_only (optional, default=False): Perform only the forward step

    timers (optional, default=None): TODO

    collect_non_loss_data: TODO

    enable_autocast (optional, default=False): If True, runs the
        forward_step_func call inside torch.autocast context

    deallocate_pipeline_outputs (optional, default=False): If True, output data
        is deallocated after the tensor is sent to the next pipeline stage.
        Helps with saving memory, does nothing when pipeline parallel is
        not used.

    no_sync_func (optional): Function that creates a context that
        suppresses asynchronous data-parallel communication. If the
        model is an instance of torch.nn.DistributedDataParallel, the
        default is to use torch.nn.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous
        gradient reductions (e.g. distributed optimizer gradient
        reduce-scatters). The function should take one argument: an
        iterable of parameters whose gradients are to be synchronized.

    param_sync_func (optional): Function that launches asynchronous
        parameter synchronizations (e.g. distributed optimizer
        parameter all-gathers). The function should take one argument:
        an iterable of parameters to be synchronized.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_using_layer_unit_test_strategy():
            # TODO (gersonkroiz): add support for interleaving supported for some out of all components
            print('get_virtual_pipeline_component_parallel_world_size(): ' + str(parallel_state.get_virtual_pipeline_component_parallel_world_size()), flush=True)
            if parallel_state.get_virtual_pipeline_component_parallel_world_size() is not None:
                print('using interleaving', flush=True)
                forward_backward_func = forward_backward_pipelining_with_interleaving
            else:
                print('not using interleaving', flush=True)
                forward_backward_func = forward_backward_pipelining_without_interleaving
        else:
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                forward_backward_func = forward_backward_pipelining_with_interleaving
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )

def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )





def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 num_microbatches,
                 input_tensor,
                 forward_data_store,
                 timers,
                 collect_non_loss_data=False,
                 autocast_dtype=torch.float,
                 enable_autocast=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    rank = torch.distributed.get_rank()
    print(f'rank {rank} | in forward_step function', flush=True)
    if timers is not None:
        timers('forward-compute', log_level=2).start()

    fwd_pass_start = datetime.now()
    rng = nvtx.start_range(message="fwd_step", color="blue")


    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    print(f'rank {rank} | before context_manager setup', flush=True)
    print(f'rank {rank} | enable_autocast: ', enable_autocast, flush=True)
    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    print(f'rank {rank} | with context_manager', flush=True)
    with context_manager:
        print(f'rank {rank} | run forward_step_func', flush=True)
        output_tensor, loss_func = forward_step_func(data_iterator, model)

    # print(f'rank {rank} | output_tensor: {output_tensor}', flush=True)
    print(f'rank {rank} | collect_non_loss_data: {collect_non_loss_data}', flush=True)
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)
    print(f'rank {rank} | after collect data', flush=True)
    if timers is not None:
        timers('forward-compute').stop()

    fwd_pass_end = datetime.now()
    nvtx.end_range(rng)

    # rank = torch.distributed.get_rank()
    # print(f'[Rank {rank}] Forward pass time schedules.py: {(fwd_pass_end - fwd_pass_start).total_seconds()}')

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)

    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(grad_scaler, input_tensor, output_tensor,
                  output_tensor_grad, model_type, timers, deallocate_pipeline_outputs=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""
    rank = torch.distributed.get_rank()
    print(f'rank {rank} | in backward_step function', flush=True)
    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    bwd_pass_start = datetime.now()
    rng = nvtx.start_range(message="bwd_step", color="red")

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        output_tensor[0] = grad_scaler(output_tensor[0])

    print(f'rank {rank} | grad_scaler is not None: {grad_scaler is not None}', flush=True)
    print(f'rank {rank} | deallocate_pipeline_outputs: {deallocate_pipeline_outputs}', flush=True)
    # print(f'rank {rank} | output_tensor[0]: {output_tensor[0]}', flush=True)
    # print(f'rank {rank} | output_tensor_grad[0]: {output_tensor_grad[0]}', flush=True)

    print(f'rank {rank} | torch.autograd.backward', flush=True)
    if deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    print(f'rank {rank} | collect grad of input tensor', flush=True)
    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)
    print(f'rank {rank} | after collect grad of input tensor', flush=True)
    print(f'rank {rank} | handle single skip connection if exists', flush=True)
    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    bwd_pass_end = datetime.now()
    nvtx.end_range(rng)
    # rank = torch.distributed.get_rank()
    # print(f'[Rank {rank}] Backward pass time schedules.py: {(bwd_pass_end - bwd_pass_start).total_seconds()}')
    print(f'rank {rank} | end of backwawrd_pass', flush=True)
    return input_tensor_grad


def forward_backward_no_pipelining(*,
                                   forward_step_func,
                                   data_iterator: Union[Iterator, List[Iterator]],
                                   model: Union[torch.nn.Module, List[torch.nn.Module]],
                                   num_microbatches: int,
                                   dtype: Optional[torch.dtype] = None,
                                   tensor_shape: Optional[Shape] = None, # unused
                                   decoder_seq_length: Optional[int] = None, # unused
                                   grad_scaler: Callable = None,
                                   sequence_parallel: bool = False, # unused
                                   overlap_p2p_comm: bool = False, # unused
                                   batch_p2p_comm: bool = True, # unused
                                   forward_only: bool = False,
                                   timers: Callable = None,
                                   collect_non_loss_data: bool = False,
                                   enable_autocast: bool = False,
                                   deallocate_pipeline_outputs: bool = False,
                                   no_sync_func: Optional[Callable] = None,
                                   grad_sync_func: Optional[Callable] = None, # unused
                                   param_sync_func: Optional[Callable] = None, # unused
                                   ):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         model, num_microbatches, input_tensor, forward_data_store,
                                         timers, collect_non_loss_data, dtype, enable_autocast)
            if not forward_only:
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 model, num_microbatches, input_tensor, forward_data_store,
                                 timers, collect_non_loss_data, dtype, enable_autocast)

    if not forward_only:
        backward_step(grad_scaler, input_tensor, output_tensor,
                      output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)

    return forward_data_store


def forward_backward_pipelining_with_interleaving(*,
                                                  forward_step_func,
                                                  data_iterator: Union[Iterator, List[Iterator]],
                                                  model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                  num_microbatches: int,
                                                  dtype: torch.dtype,
                                                  tensor_shape: Shape,
                                                  decoder_seq_length: Optional[int] = None,
                                                  grad_scaler: Callable = None,
                                                  sequence_parallel: bool = False,
                                                  overlap_p2p_comm: bool = False,
                                                  batch_p2p_comm: bool = True,
                                                  forward_only: bool = False,
                                                  timers: Callable = None,
                                                  collect_non_loss_data: bool = False,
                                                  enable_autocast: bool = False,
                                                  deallocate_pipeline_outputs: bool = False,
                                                  no_sync_func: Optional[Callable] = None,
                                                  grad_sync_func: Optional[Callable] = None,
                                                  param_sync_func: Optional[Callable] = None,
                                                  ):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), \
        "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    if overlap_p2p_comm and batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
    print(f'################## in schedules, overlap_p2p_comm: {overlap_p2p_comm}, batch_p2p_comm: {batch_p2p_comm} ##################', flush=True)
    # Disable async grad reductions
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):
        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack
        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    
    pipeline_component_parallel_size, pipeline_component_parallel_rank = None, None
    if parallel_state.get_using_layer_unit_test_strategy():
        pipeline_component_parallel_size = parallel_state.get_pipeline_component_parallel_world_size()
        pipeline_component_parallel_rank = parallel_state.get_pipeline_component_parallel_rank()

    rank = torch.distributed.get_rank()

    print(f'rank {rank} | in forward_backward_pipelining_with_interleaving', flush=True)
    print(f'rank {rank} | pipeline_model_parallel_size: {pipeline_model_parallel_size}', flush=True)
    print(f'rank {rank} | pipeline_model_parallel_rank: {pipeline_model_parallel_rank}', flush=True)
    print(f'rank {rank} | pipeline_component_parallel_size: {pipeline_component_parallel_size}', flush=True)
    print(f'rank {rank} | pipeline_component_rank: {pipeline_component_parallel_rank}', flush=True)

    # TODO (gersonkroiz): need to adjust for non-uniform data parallelism
    if num_microbatches % pipeline_model_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_model_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    if parallel_state.get_using_layer_unit_test_strategy():
        if num_microbatches % pipeline_component_parallel_size != 0:
            msg = f'number of microbatches ({num_microbatches}) is not divisible by '
            msg += f'pipeline-component-parallel-size ({pipeline_component_parallel_size}) '
            msg += 'when using interleaved schedule'
            raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError("Interleaving is not supported with a different decoder sequence length.")

    # TODO (gersonkroiz): check this
    print(f'rank {rank} | using sequence_parallel: {sequence_parallel == True}', flush=True)
    
    if sequence_parallel:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // parallel_state.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    # Compute number of warmup and remaining microbatches.
    print(f'rank {rank} | compute number of warmup and remaining microbatches', flush=True)
    num_model_chunks = len(model)
    num_component_chunks = num_model_chunks
    print(f'rank {rank} | num_model_chunks: {num_model_chunks}', flush=True)
    print(f'rank {rank} | num_microbatches: {num_microbatches}', flush=True)
    total_num_microbatches = num_microbatches * num_model_chunks
    print(f'rank {rank} | total_num_microbatches: {total_num_microbatches}', flush=True)
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
        print(f'rank {rank} | num_warmup_microbatches calculation 1')
        # if parallel_state.get_using_layer_unit_test_strategy():
                # if not parallel_state.is_rank_in_first_component():
                    # num_warmup_microbatches -= 1
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_model_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_model_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            print(f'rank {rank} | num_warmup_microbatches calculation 2')
            all_warmup_microbatches = True
        else:
            if parallel_state.get_using_layer_unit_test_strategy():
                num_warmup_microbatches = \
                    (pipeline_component_parallel_size - pipeline_component_parallel_rank - 1) * 2
                num_warmup_microbatches += (
                    num_model_chunks - 1) * pipeline_component_parallel_size
                
                # num_warmup_microbatches = \
                #     (pipeline_component_parallel_size - pipeline_component_parallel_rank - 1) * 2
                # num_warmup_microbatches += (
                #     num_model_chunks - 1) * pipeline_component_parallel_size
                if not parallel_state.is_rank_in_last_component():
                    num_warmup_microbatches += 2 * (num_model_chunks * parallel_state.get_pipeline_component_parallel_world_size('response'))
                    # num_warmup_microbatches += num_model_chunks * parallel_state.get_pipeline_component_parallel_world_size('test')
                if parallel_state.is_rank_in_first_component():
                    num_warmup_microbatches += 2 * (num_model_chunks * parallel_state.get_pipeline_component_parallel_world_size('test'))
                # d = {0:31,1:29,2:27,3:19,4:17,5:15,6:7,7:5,8:3}
                # d = {0:31,1:29,2:27,3:19,4:17,5:15,6:7,7:5,8:3}
                # d = {0:,1:,2:,3:,4:,5:,6:,7:3+2*3*4,8:9,9:7,10:5,11:3}
                # d = {
                #     0:46,
                #     1:44,
                #     2:42,
                #     3:4+9+9+6,
                #     4:2+9+9+6,
                #     5:9+9+6,
                #     6:10,
                #     7:8,
                #     8:6,
                # }
                
                # 4
                # 2+4
                # 4+4
                # 6+4
                # 8+4
                
                
                # 2 has to wait for 2nd component loop 2.5 times, and then another 2 times
                
                # in last component, first 1F1B is last time to take from prev component for that iteration
                
                # num_warmup_microbatches = d[rank]
            else:
                num_warmup_microbatches = \
                    (pipeline_model_parallel_size - pipeline_model_parallel_rank - 1) * 2
                num_warmup_microbatches += (
                    num_model_chunks - 1) * pipeline_model_parallel_size
                
                # num_model_chunks = 2
                # pipeline_model_parallel_size = 3
                # pipeilne_component_parallel_size = 3
                
                # first rank would have 25
                # on last rank, num_warmup_mb = 9
            num_warmup_microbatches = min(num_warmup_microbatches,
                                        total_num_microbatches)
            print(f'rank {rank} | num_warmup_microbatches calculation 3')
            
    num_microbatches_remaining = \
        total_num_microbatches - num_warmup_microbatches

    print(f'rank {rank} | num_warmup_microbatches: {num_warmup_microbatches}', flush=True)
    print(f'rank {rank} | num_microbatches_remaining: {num_microbatches_remaining}', flush=True)
    print(f'rank {rank} | num_model_chunks: {num_model_chunks}', flush=True)
    # Synchronize params for first two model chunks
    if param_sync_func is not None:
        param_sync_func(model[0].parameters())
        param_sync_func(model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        print(f'rank {rank} | in get_model_chunk_id, microbatch_id: {microbatch_id}')
        if parallel_state.get_using_layer_unit_test_strategy():
            microbatch_id_in_group = microbatch_id % (pipeline_component_parallel_size * num_model_chunks)
            print(f'rank {rank} | microbatch_id_in_group: {microbatch_id_in_group}')
            model_chunk_id = microbatch_id_in_group // pipeline_component_parallel_size
            print(f'rank {rank} | model_chunk_id: {model_chunk_id}')
        else:
            microbatch_id_in_group = microbatch_id % (pipeline_model_parallel_size * num_model_chunks)
            model_chunk_id = microbatch_id_in_group // pipeline_model_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
            print(f'rank {rank} | not forward, new model_chunk_id: {model_chunk_id}')
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        print(f'rank {rank} | in is_first_microbatch_for_model_chunk', flush=True)
        if parallel_state.get_using_layer_unit_test_strategy():
            microbatch_group_size = pipeline_component_parallel_size * num_model_chunks
            num_microbatch_groups = total_num_microbatches // microbatch_group_size
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            if microbatch_group_id == 0:
                return microbatch_id_in_group % pipeline_component_parallel_size == 0
            else:
                return False
        else:
            microbatch_group_size = pipeline_model_parallel_size * num_model_chunks
            num_microbatch_groups = total_num_microbatches // microbatch_group_size
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            if microbatch_group_id == 0:
                return microbatch_id_in_group % pipeline_model_parallel_size == 0
            else:
                return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        print(f'rank {rank} | in is_last_microbatch_for_model_chunk', flush=True)
        if parallel_state.get_using_layer_unit_test_strategy():
            microbatch_group_size = pipeline_component_parallel_size * num_model_chunks
            num_microbatch_groups = total_num_microbatches // microbatch_group_size
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            if microbatch_group_id == num_microbatch_groups - 1:
                return microbatch_id_in_group % pipeline_component_parallel_size == pipeline_component_parallel_size - 1
            else:
                return False
        else:
            microbatch_group_size = pipeline_model_parallel_size * num_model_chunks
            num_microbatch_groups = total_num_microbatches // microbatch_group_size
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            if microbatch_group_id == num_microbatch_groups - 1:
                return microbatch_id_in_group % pipeline_model_parallel_size == pipeline_model_parallel_size - 1
            else:
                return False

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        print(f'rank {rank} | in forward_step_helper', flush=True)
        print(f'rank {rank} | microbatch_id: {microbatch_id}', flush=True)
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        print(f'rank {rank} | ---- model_chunk_id for forward: {model_chunk_id}', flush=True)
        if parallel_state.get_using_layer_unit_test_strategy():
            parallel_state.set_virtual_pipeline_component_parallel_rank(model_chunk_id)
        else:
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if param_sync_func is not None:
            if parallel_state.get_using_layer_unit_test_strategy():
                param_sync_microbatch_id = microbatch_id + pipeline_component_parallel_rank
            else:
                param_sync_microbatch_id = microbatch_id + pipeline_model_parallel_rank
            if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(param_sync_microbatch_id):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    param_sync_func(model[param_sync_chunk_id].parameters())

        # forward step
        if parallel_state.is_pipeline_first_stage():
            print(f'rank {rank} | len(input_tensors[model_chunk_id]): {len(input_tensors[model_chunk_id])}', flush=True)
            print(f'rank {rank} | len(output_tensors[model_chunk_id]): {len(output_tensors[model_chunk_id])}', flush=True)
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        print(f'rank {rank} | model[model_chunk_id]: {model[model_chunk_id]}', flush=True)
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     num_microbatches,
                                     input_tensor,
                                     forward_data_store,
                                     timers,
                                     collect_non_loss_data,
                                     dtype,
                                     enable_autocast)
        # print(f'rank {rank} | fwd_step input_tensor: {input_tensor}', flush=True)
        # print(f'rank {rank} | fwd_step output_tensor: {output_tensor}', flush=True)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        print(f'rank {rank} | in backward_step_helper', flush=True)
        print(f'rank {rank} | microbatch_id: {microbatch_id}', flush=True)
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        print(f'rank {rank} | ---- model_chunk_id for backward: {model_chunk_id}', flush=True)
        if parallel_state.get_using_layer_unit_test_strategy():
            parallel_state.set_virtual_pipeline_component_parallel_rank(model_chunk_id)
        else:
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        print(f'rank {rank} | launch grad_sync_func', flush=True)
        # launch grad synchronization (default)
        if grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)
        print(f'rank {rank} | after launch grad_sync_func', flush=True)
        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        print(f'rank {rank} | backward_step', flush=True)
        print(f'rank {rank} | grad_scaler: {grad_scaler}', flush=True)
        print(f'rank {rank} | input_tensor: {input_tensor}', flush=True)
        print(f'rank {rank} | output_tensor: {output_tensor}', flush=True)
        print(f'rank {rank} | output_tensor_grad: {output_tensor_grad}', flush=True)
        print(f'rank {rank} | model_type: {model_type}', flush=True)
        print(f'rank {rank} | timers: {timers}', flush=True)
        print(f'rank {rank} | deallocate_pipeline_outputs: {deallocate_pipeline_outputs}', flush=True)
        input_tensor_grad = \
            backward_step(grad_scaler,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad,
                          model_type,
                          timers,
                          deallocate_pipeline_outputs)

        # print(f'rank {rank} | bkwd_step input_tensor: {input_tensor}', flush=True)
        # print(f'rank {rank} | bkwd_step input_tensor: {output_tensor}', flush=True)
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if grad_sync_func is not None:
            if parallel_state.get_using_layer_unit_test_strategy():
                grad_sync_microbatch_id = microbatch_id - pipeline_component_parallel_rank
            else:
                grad_sync_microbatch_id = microbatch_id - pipeline_model_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(grad_sync_microbatch_id):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                grad_sync_func(model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad
    
    def within_component_helper():
        # calculate within_component
        # within_component = False if recv_prev/send_prev from/to prev component (check that prev component exists) 
        # within_component = False if recv_next/send_next from/to next component (check that next component exists)
        # within_component = True in other cases
        within_component = True
        if parallel_state.is_pipeline_component_last_stage() and not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            within_component = False
        elif parallel_state.is_pipeline_component_first_stage() and not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            within_component = False
        return within_component

    # forward_backward_pipelining_with_interleaving when using LayerUnitTestStrategy()
    if parallel_state.get_using_layer_unit_test_strategy():    
        # Run warmup forward passes.
        print(f'rank {rank} | run warmup forward passes', flush=True)
        parallel_state.set_virtual_pipeline_component_parallel_rank(0)
        # within_component is False, since this is warmup, go through full model pipeline parallel group
        # TODO (gersonkroiz): check whether we need warmup within a component's interleaving schedule
        input_tensors[0].append(
            p2p_communication.recv_forward(tensor_shape,
                                        dtype=dtype,
                                        batch_p2p_comm=batch_p2p_comm,
                                        timers=timers,
                                        within_component=False))

        print(f'rank {rank} | before warmup, input_tensors[0]: {input_tensors[0]}', flush=True)
        
        fwd_wait_handles = None
        bwd_wait_handles = None

        print(f'rank {rank} | num_warmup_microbatches: {num_warmup_microbatches}', flush=True)
        for k in range(num_warmup_microbatches):
            print(f'----rank {rank} | beginning of warmup iter {k}/{num_warmup_microbatches}----', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_first_stage(): {parallel_state.is_pipeline_first_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_last_stage(): {parallel_state.is_pipeline_last_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_component_first_stage(ignore_virtual=True): {parallel_state.is_pipeline_component_first_stage(ignore_virtual=True)}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_component_first_stage(): {parallel_state.is_pipeline_component_first_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_component_last_stage(ignore_virtual=True): {parallel_state.is_pipeline_component_last_stage(ignore_virtual=True)}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_component_last_stage(): {parallel_state.is_pipeline_component_last_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_rank(): {parallel_state.get_virtual_pipeline_model_parallel_rank()}', flush=True)
            print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_world_size(): {parallel_state.get_virtual_pipeline_model_parallel_world_size()}', flush=True)

            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()
            print(f'rank {rank} | forward_step_helper({k})', flush=True)
            output_tensor = forward_step_helper(k)

            # Determine if tensor should be received from previous stage.
            next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
            print(f'rank {rank} | ---- next_forward_model_chunk_id: {next_forward_model_chunk_id}', flush=True)
            recv_prev = True
            # NOTE (gersonkroiz): leave as is
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False
            if k == (total_num_microbatches - 1):
                recv_prev = False

            
            # After the first component, each rank will need to receive a tensor before iterating through warmup microbatches.
            # the virtual_pipeline_component_parallel_rank will need to be adjusted before send/recv tensors to componensate for
            # the 1 extra recv before the iteration loop.
            print(f'rank {rank} | changing virtual rank before p2p comm: {not parallel_state.is_rank_in_first_component() and not parallel_state.is_pipeline_component_last_stage(ignore_virtual=True)}')
            # if not parallel_state.is_rank_in_first_component() and not parallel_state.is_pipeline_component_last_stage(ignore_virtual=True):
            if not parallel_state.is_rank_in_first_component() and not parallel_state.is_pipeline_component_last_stage(ignore_virtual=True):
                parallel_state.set_virtual_pipeline_component_parallel_rank(next_forward_model_chunk_id) 

            # Don't send tensor downstream if on last stage.
            # print(f'rank {rank} | should send tensor downstream? {parallel_state.is_pipeline_last_stage()}', flush=True)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if not overlap_p2p_comm:
                # check if last microbatch
                if k == (num_warmup_microbatches - 1) and not forward_only and \
                        not all_warmup_microbatches:
                    print(f'rank {rank} | last microbatch', flush=True)
                    input_tensor_grad = None
                    recv_next = True
                    # NOTE (gersonkroiz): leave as is
                    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False
                    input_tensor, output_tensor_grad = \
                        p2p_communication.send_forward_backward_recv_forward_backward(
                            output_tensor, input_tensor_grad,
                            recv_prev=recv_prev, recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            dtype=dtype,
                            batch_p2p_comm=batch_p2p_comm,
                            timers=timers,
                            within_component=within_component_helper())
                    if parallel_state.is_pipeline_first_stage():
                        print(f'rank {rank} | (0) p2p_communication.send_forward_backward_recv_forward_backward from rank {parallel_state.get_pipeline_component_parallel_prev_ranks()} to none (first rank)', flush=True)
                    elif parallel_state.is_pipeline_last_stage():
                        print(f'rank {rank} | (0) p2p_communication.send_forward_backward_recv_forward_backward from none (last rank) to rank {parallel_state.get_pipeline_component_parallel_next_ranks()}', flush=True)
                    else:
                        print(f'rank {rank} | (0) p2p_communication.send_forward_backward_recv_forward_backward from rank {parallel_state.get_pipeline_component_parallel_prev_ranks()} to rank {parallel_state.get_pipeline_component_parallel_next_ranks()}', flush=True)
                    print(f'rank {rank} | (0) output_tensor: {output_tensor}', flush=True)
                    print(f'rank {rank} | (0) input_tensor_grad: {input_tensor_grad}', flush=True)
                    print(f'rank {rank} | (0) input_tensor: {input_tensor}', flush=True)
                    print(f'rank {rank} | (0) output_tensor_grad: {output_tensor_grad}', flush=True)
                        
                    output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
                else:
                    print(f'rank {rank} | not last microbatch', flush=True)
                    rank = torch.distributed.get_rank()
                    # TODO (gersonkroiz): check this
                    input_tensor = \
                        p2p_communication.send_forward_recv_forward(
                            output_tensor, recv_prev=recv_prev,
                            tensor_shape=tensor_shape,
                            dtype=dtype,
                            batch_p2p_comm=batch_p2p_comm,
                            timers=timers,
                            within_component=within_component_helper())
                    if recv_prev and output_tensor != None: 
                        print(f'rank {rank} | (1) send_forward_recv_forward (both)', flush=True)
                    elif recv_prev:
                        print(f'rank {rank} | (1) send_forward_recv_forward (only recv)', flush=True)
                    elif output_tensor != None:
                        print(f'rank {rank} | (1) send_forward_recv_forward (only send)', flush=True)
                    else:
                        print(f'rank {rank} | (1) send_forward_recv_forward (nothing)', flush=True)
                    print(f'rank {rank} | (1) output_tensor: {output_tensor}', flush=True)
                    print(f'rank {rank} | (1) input_tensor: {input_tensor}', flush=True)
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            else:
                rank = torch.distributed.get_rank()
                # TODO (gersonkroiz): check this
                print(f'rank {rank} | (2) output_tensor: {output_tensor}', flush=True)
                input_tensor, fwd_wait_handles = \
                    p2p_communication.send_forward_recv_forward(
                        output_tensor, recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers,
                        overlap_p2p_comm=True,
                        within_component=within_component_helper())

                if recv_prev and output_tensor != None: 
                    print(f'rank {rank} | (2) send_forward_recv_forward (both)', flush=True)
                elif recv_prev:
                    print(f'rank {rank} | (2) send_forward_recv_forward (only recv)', flush=True)
                elif output_tensor != None:
                    print(f'rank {rank} | (2) send_forward_recv_forward (only send)', flush=True)
                else:
                    print(f'rank {rank} | (2) send_forward_recv_forward (nothing)', flush=True)
                print(f'rank {rank} | (2) output_tensor: {output_tensor}', flush=True)
                print(f'rank {rank} | (2) input_tensor: {input_tensor}', flush=True)

                # check if last microbatch
                print(f'rank {rank} | check if last microbatch: {k == (num_warmup_microbatches - 1) and not forward_only and not all_warmup_microbatches}', flush=True)
                if k == (num_warmup_microbatches - 1) and not forward_only and \
                        not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    
                    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False
                    print(f'rank {rank} | within_component_helper before backward: {within_component_helper()}')
                    print(f'rank {rank} | (3) input_tensor_grad: {input_tensor_grad}', flush=True)

                    output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        batch_p2p_comm=batch_p2p_comm,
                        dtype=dtype,
                        timers=timers,
                        overlap_p2p_comm=True,
                        within_component=False)
                    if recv_next and output_tensor_grad != None: 
                        print(f'rank {rank} | (3) send_backward_recv_backward (both)', flush=True)
                    elif recv_next:
                        print(f'rank {rank} | (3) send_backward_recv_backward (only recv)', flush=True)
                    elif output_tensor_grad != None:
                        print(f'rank {rank} | (3) send_backward_recv_backward (only send)', flush=True)
                    else:
                        print(f'rank {rank} | (3) send_backward_recv_backward (nothing)', flush=True)
                    print(f'rank {rank} | (3) output_tensor_grad: {output_tensor_grad}', flush=True)
                    print(f'rank {rank} | (3) input_tensor_grad: {input_tensor_grad}', flush=True)

                    output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
                input_tensors[next_forward_model_chunk_id].append(input_tensor)

            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)
        print(f'------rank {rank} | warmup complete------', flush=True)
        # print(f'rank {rank} | before barrier', flush=True)
        # torch.distributed.barrier()
        # print(f'rank {rank} | after barrier', flush=True)
        print(f'rank {rank} | run 1F1B in steady state, num_microbatches_remaining={num_microbatches_remaining}', flush=True)
        print(f'rank {rank} | overlap_p2p_comm: {overlap_p2p_comm}', flush=True)
        # Run 1F1B in steady state.
        for k in range(num_microbatches_remaining):
            print(f'----rank {rank} | beginning of 1F1B iter {k}/{num_microbatches_remaining}----', flush=True)
            # Forward pass.
            forward_k = k + num_warmup_microbatches
            print(f'rank {rank} | forward_k: {forward_k}', flush=True)

            if overlap_p2p_comm:
                print(f'rank {rank} | p2p overlap', flush=True)
                if fwd_wait_handles is not None:
                    for req in fwd_wait_handles:
                        req.wait()

                deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

                print(f'rank {rank} | call forward_step_helper({forward_k})', flush=True)
                output_tensor = forward_step_helper(forward_k)

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                print(f'rank {rank} | forward_model_chunk_id: {forward_model_chunk_id}', flush=True)
                parallel_state.set_virtual_pipeline_component_parallel_rank(forward_model_chunk_id)
                
                # Last virtual stage no activation tensor to send
                print(f'rank {rank} | is last component stage? {parallel_state.is_pipeline_component_last_stage()}', flush=True)
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_component_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(
                        forward_k - (pipeline_component_parallel_size - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                                    forward=True)
                print(f'rank {rank} | next_forward_model_chunk_id: {next_forward_model_chunk_id}', flush=True)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                # print(f'rank {rank} | last iteration: {k == (num_microbatches_remaining - 1)}', flush=True)
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False
                print(f'rank {rank} | should recieve from prev? {recv_prev}', flush=True)
                
                # After the first component, each rank will need to receive a tensor before iterating through warmup microbatches.
                # the virtual_pipeline_component_parallel_rank will need to be adjusted before send/recv tensors to componensate for
                # the 1 extra recv before the iteration loop.
                print(f'rank {rank} | changing virtual rank before p2p comm: {not parallel_state.is_rank_in_first_component()}')
                if not parallel_state.is_rank_in_first_component() and not parallel_state.is_pipeline_component_last_stage(ignore_virtual=True):
                    parallel_state.set_virtual_pipeline_component_parallel_rank(next_forward_model_chunk_id) 
                
                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage. 
                print(f'rank {rank} | output_tensor: {output_tensor}', flush=True)
                input_tensor, fwd_wait_handles = \
                    p2p_communication.send_forward_recv_forward(
                        output_tensor, recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers,
                        overlap_p2p_comm=True,
                        within_component = within_component_helper())
                print(f'rank {rank} | p2p_communication.send_forward_recv_forward', flush=True)
                print(f'rank {rank} | output_tensor: {output_tensor}', flush=True)
                print(f'rank {rank} | input_tensor: {input_tensor}', flush=True)
                # assert fwd_wait_handles is not None

                if bwd_wait_handles is not None:
                    for req in bwd_wait_handles:
                        req.wait()

                # Backward pass.
                print(f'rank {rank} | backward pass', flush=True)
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                print(f'rank {rank} | backward_model_chunk_id: {backward_model_chunk_id}', flush=True)
                parallel_state.set_virtual_pipeline_component_parallel_rank(backward_model_chunk_id)

                # First virtual stage no activation gradient tensor to send
                if parallel_state.is_pipeline_first_stage():
                    print(f'rank {rank} | is first virtual stage, no activation grad tensor to send', flush=True)
                    input_tensor_grad = None
                else:
                    print(f'rank {rank} | is not first virtual stage, activation grad tensor to send', flush=True)

                # Determine if the current virtual stage has an activation gradient tensor to receive
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_component_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (pipeline_component_parallel_size - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k + 1, forward=False
                    )

                print(f'rank {rank} | next_backward_model_chunk_id: {next_backward_model_chunk_id}', flush=True)

                # After the first component, each rank will need to receive a tensor before iterating through warmup microbatches.
                # the virtual_pipeline_component_parallel_rank will need to be adjusted before send/recv tensors to componensate for
                # the 1 extra recv before the iteration loop.
                print(f'rank {rank} | changing virtual rank before p2p comm: {not parallel_state.is_rank_in_last_component() and not parallel_state.is_pipeline_component_first_stage(ignore_virtual=True)}')
                if not parallel_state.is_rank_in_last_component() and not parallel_state.is_pipeline_component_first_stage(ignore_virtual=True):
                    parallel_state.set_virtual_pipeline_component_parallel_rank(next_backward_model_chunk_id) 

                print(f'rank {rank} | input_tensor_grad: {input_tensor_grad}', flush=True)

                output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                    overlap_p2p_comm=True,
                    within_component=within_component_helper())
                print(f'rank {rank} | p2p_communication.send_backward_recv_backward', flush=True)
                print(f'rank {rank} | output_tensor_grad: {output_tensor_grad}', flush=True)
                print(f'rank {rank} | input_tensor_grad: {input_tensor_grad}', flush=True)

                if recv_prev:
                    print(f'rank {rank} | will recv from prev ', flush=True)
                else:
                    print(f'rank {rank} | will not recv from prev ', flush=True)

                if output_tensor is not None:
                    print(f'rank {rank} | will send to next ', flush=True)
                else:
                    print(f'rank {rank} | will not send to next ', flush=True)

                if recv_next:
                    print(f'rank {rank} | will recv from next ', flush=True)
                else:
                    print(f'rank {rank} | will not recv from next ', flush=True)
                    
                if input_tensor_grad is not None:
                    print(f'rank {rank} | will send to prev ', flush=True)
                else:
                    print(f'rank {rank} | will not send to prev ', flush=True)     

            else: # no p2p overlap
                print(f'rank {rank} | no p2p overlap', flush=True)
                print(f'rank {rank} | forward_step_helper', flush=True)
                output_tensor = forward_step_helper(forward_k)

                # Backward pass.
                backward_k = k
                print(f'rank {rank} | backward_step_helper', flush=True)
                input_tensor_grad = backward_step_helper(backward_k)

                # Send output_tensor and input_tensor_grad, receive input_tensor
                # and output_tensor_grad.

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                print(f'rank {rank} | ---- forward_model_chunk_id: {forward_model_chunk_id}', flush=True)
                parallel_state.set_virtual_pipeline_component_parallel_rank(forward_model_chunk_id)
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = None
                print(f'rank {rank} | do we need to send foward? {output_tensor != None}', flush=True)

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                print(f'rank {rank} | ---- backward_model_chunk_id: {backward_model_chunk_id}', flush=True)
                parallel_state.set_virtual_pipeline_component_parallel_rank(backward_model_chunk_id)
                if parallel_state.is_pipeline_first_stage():
                    input_tensor_grad = None
                print(f'rank {rank} | do we need to send backward? {input_tensor_grad != None}', flush=True)

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                print(f'rank {rank} | forward_k: {forward_k}', flush=True)
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_component_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(
                        forward_k - (pipeline_component_parallel_size - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                                    forward=True)
                print(f'rank {rank} | ---- next_forward_model_chunk_id: {next_forward_model_chunk_id}', flush=True)

                recv_next = True
                print(f'rank {rank} | backward_k: {backward_k}', flush=True)
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_component_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (pipeline_component_parallel_size - 1), forward=False)
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                                    forward=False)
                print(f'rank {rank} | ---- next_backward_model_chunk_id: {next_backward_model_chunk_id}', flush=True)

                # if not parallel_state.is_rank_in_first_component() and not parallel_state.is_pipeline_component_last_stage(ignore_virtual=True):
                #     parallel_state.set_virtual_pipeline_component_parallel_rank(next_forward_model_chunk_id) 

                # if not parallel_state.is_rank_in_last_component() and not parallel_state.is_pipeline_component_first_stage(ignore_virtual=True):
                #     parallel_state.set_virtual_pipeline_component_parallel_rank(next_backward_model_chunk_id) 

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False
                # print(f'rank {rank} | is last iteration? {k == (num_microbatches_remaining - 1)}', flush=True)


                if recv_prev:
                    print(f'rank {rank} | will recv from prev ', flush=True)
                else:
                    print(f'rank {rank} | will not recv from prev ', flush=True)

                if output_tensor is not None:
                    print(f'rank {rank} | will send to next ', flush=True)
                else:
                    print(f'rank {rank} | will not send to next ', flush=True)

                if recv_next:
                    print(f'rank {rank} | will recv from next ', flush=True)
                else:
                    print(f'rank {rank} | will not recv from next ', flush=True)
                    
                if input_tensor_grad is not None:
                    print(f'rank {rank} | will send to prev ', flush=True)
                else:
                    print(f'rank {rank} | will not send to prev ', flush=True)        
            
                
                # Communicate tensors.
                input_tensor, output_tensor_grad = \
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers,
                        within_component=within_component_helper())

                print(f'rank {rank} | send_forward_backward_recv_forward_backward', flush=True)
                print(f'rank {rank} | output_tensor: {output_tensor}', flush=True)
                print(f'rank {rank} | input_tensor_grad: {input_tensor_grad}', flush=True)
                print(f'rank {rank} | input_tensor: {input_tensor}', flush=True)
                print(f'rank {rank} | output_tensor_grad: {output_tensor_grad}', flush=True)

                deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(
                    output_tensor_grad)

        deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

        print(f'rank {rank} | after 1F1B cooldown backward passes', flush=True)
        # Run cooldown backward passes (flush out pipeline).
        if not forward_only:
            if overlap_p2p_comm and bwd_wait_handles is not None:
                for wait_handle in bwd_wait_handles:
                    wait_handle.wait()

            if all_warmup_microbatches:
                output_tensor_grads[num_model_chunks-1].append(
                    p2p_communication.recv_backward(tensor_shape,
                                                    dtype=dtype,
                                                    batch_p2p_comm=batch_p2p_comm,
                                                    timers=timers))
                print(f'rank {rank} | p2p_communication.recv_backward', flush=True)
                print(f'rank {rank} | output_tensor_grad: {output_tensor_grads[num_model_chunks-1][-1]}', flush=True)

            print(f'rank {rank} | num_microbatches_remaining: {num_microbatches_remaining}', flush=True)
            print(f'rank {rank} | total_num_microbatches: {total_num_microbatches}', flush=True)
            for k in range(num_microbatches_remaining, total_num_microbatches):
                print(f'----rank {rank} | cooldown backwawrd passes {k}/{num_warmup_microbatches}----', flush=True)
                print(f'rank {rank} | parallel_state.is_pipeline_first_stage(): {parallel_state.is_pipeline_first_stage()}', flush=True)
                print(f'rank {rank} | parallel_state.is_pipeline_last_stage(): {parallel_state.is_pipeline_last_stage()}', flush=True)
                print(f'rank {rank} | parallel_state.is_pipeline_component_first_stage(ignore_virtual=True): {parallel_state.is_pipeline_component_first_stage(ignore_virtual=True)}', flush=True)
                print(f'rank {rank} | parallel_state.is_pipeline_component_first_stage(): {parallel_state.is_pipeline_component_first_stage()}', flush=True)
                print(f'rank {rank} | parallel_state.is_pipeline_component_last_stage(ignore_virtual=True): {parallel_state.is_pipeline_component_last_stage(ignore_virtual=True)}', flush=True)
                print(f'rank {rank} | parallel_state.is_pipeline_component_last_stage(): {parallel_state.is_pipeline_component_last_stage()}', flush=True)
                print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_rank(): {parallel_state.get_virtual_pipeline_model_parallel_rank()}', flush=True)
                print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_world_size(): {parallel_state.get_virtual_pipeline_model_parallel_world_size()}', flush=True)

                print(f'rank {rank} | backward_step_helper({k})', flush=True)
                input_tensor_grad = backward_step_helper(k)
                next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
                print(f'rank {rank} | next_backward_model_chunk_id: {next_backward_model_chunk_id}', flush=True)

                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (num_model_chunks - 1):
                        recv_next = False
                if k == (total_num_microbatches - 1):
                    print(f'rank {rank} | last microbatch', flush=True)
                    recv_next = False
                
                # After the first component, each rank will need to receive a tensor before iterating through warmup microbatches.
                # the virtual_pipeline_component_parallel_rank will need to be adjusted before send/recv tensors to componensate for
                # the 1 extra recv before the iteration loop.
                print(f'rank {rank} | changing virtual rank before p2p comm: {not parallel_state.is_rank_in_last_component() and not parallel_state.is_pipeline_component_first_stage(ignore_virtual=True)}')
                if not parallel_state.is_rank_in_last_component() and not parallel_state.is_pipeline_component_first_stage(ignore_virtual=True):
                    parallel_state.set_virtual_pipeline_component_parallel_rank(next_backward_model_chunk_id) 

                print(f'rank {rank} | recv_next: {recv_next}', flush=True)
                output_tensor_grads[next_backward_model_chunk_id].append(
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers,
                        within_component=within_component_helper()))
                print(f'rank {rank} | p2p_communication.send_backward_recv_backward', flush=True)
                if recv_next and input_tensor_grad != None: 
                    print(f'rank {rank} | (1) send_backward_recv_backward (both)', flush=True)
                elif recv_next:
                    print(f'rank {rank} | (1) send_backward_recv_backward (only recv)', flush=True)
                elif input_tensor_grad != None:
                    print(f'rank {rank} | (1) send_backward_recv_backward (only send)', flush=True)
                else:
                    print(f'rank {rank} | (1) send_backward_recv_backward (nothing)', flush=True)
                print(f'rank {rank} | (1) output_tensor_grads[next_backward_model_chunk_id]: {output_tensor_grads[next_backward_model_chunk_id]}', flush=True)
                print(f'rank {rank} | (1) input_tensor_grad: {input_tensor_grad}', flush=True)

        print(f'------rank {rank} | cooldown complete------', flush=True)

        # Launch any remaining grad reductions
        print(f'rank {rank} | launch remaining grad reductions', flush=True)
        enable_grad_sync()
        if grad_sync_func is not None:
            params = []
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    params.extend(model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
            if params:
                grad_sync_func(params)

        # print(f'rank {rank} | before barrier', flush=True)
        # torch.distributed.barrier()
        # print(f'rank {rank} | after barrier', flush=True)

        print(f'################## rank {rank} | leaving forward_backward_pipelining_with_interleaving ##################', flush=True)
        
        return forward_data_store

    # forward_backward_pipelining_with_interleaving when not using LayerUnitTestStrategy()
    else:
        # Run warmup forward passes.
        print(f'rank {rank} | run warmup forward passes', flush=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        input_tensors[0].append(
            p2p_communication.recv_forward(tensor_shape,
                                        dtype=dtype,
                                        batch_p2p_comm=batch_p2p_comm,
                                        timers=timers))
        print(f'rank {rank} | p2p_communication.recv_forward from rank {parallel_state.get_pipeline_model_parallel_prev_ranks()}', flush=True)
        print(f'rank {rank} | output_tensor_grad: {input_tensors[0][-1]}', flush=True)

        fwd_wait_handles = None
        bwd_wait_handles = None

        print(f'rank {rank} | num_warmup_microbatches: {num_warmup_microbatches}', flush=True)
        for k in range(num_warmup_microbatches):
            print(f'----rank {rank} | beginning of warmup iter {k}/{num_warmup_microbatches}----', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_first_stage(ignore_virtual=True): {parallel_state.is_pipeline_first_stage(ignore_virtual=True)}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_first_stage(): {parallel_state.is_pipeline_first_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_last_stage(ignore_virtual=True): {parallel_state.is_pipeline_last_stage(ignore_virtual=True)}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_last_stage(): {parallel_state.is_pipeline_last_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_rank(): {parallel_state.get_virtual_pipeline_model_parallel_rank()}', flush=True)
            print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_world_size(): {parallel_state.get_virtual_pipeline_model_parallel_world_size()}', flush=True)

            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            output_tensor = forward_step_helper(k)

            # Determine if tensor should be received from previous stage.
            next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
            print(f'rank {rank} | next_forward_model_chunk_id: ' + str(next_forward_model_chunk_id))
            recv_prev = True
            
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False
            if k == (total_num_microbatches - 1):
                recv_prev = False

            # Don't send tensor downstream if on last stage.
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if not overlap_p2p_comm:
                # check if last microbatch
                if k == (num_warmup_microbatches - 1) and not forward_only and \
                        not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False
                    print(f'rank {rank} | (0) p2p_communication.send_forward_backward_recv_forward_backward in rank {rank}', flush=True)
                    input_tensor, output_tensor_grad = \
                        p2p_communication.send_forward_backward_recv_forward_backward(
                            output_tensor, input_tensor_grad,
                            recv_prev=recv_prev, recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            dtype=dtype,
                            batch_p2p_comm=batch_p2p_comm,
                            timers=timers)
                    print(f'rank {rank} | (0) output_tensor: {output_tensor}', flush=True)
                    print(f'rank {rank} | (0) input_tensor: {input_tensor}', flush=True)
                    output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
                else:
                    rank = torch.distributed.get_rank()
                    print(f'rank {rank} | (1) p2p_communication.send_forward_recv_forward in rank {rank}', flush=True)
                    input_tensor = \
                        p2p_communication.send_forward_recv_forward(
                            output_tensor, recv_prev=recv_prev,
                            tensor_shape=tensor_shape,
                            dtype=dtype,
                            batch_p2p_comm=batch_p2p_comm,
                            timers=timers)
                    print(f'rank {rank} | (1) output_tensor: {output_tensor}', flush=True)
                    print(f'rank {rank} | (1) input_tensor: {input_tensor}', flush=True)
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            else:
                rank = torch.distributed.get_rank()
                print(f'rank {rank} | (2) p2p_communication.send_forward_recv_forward in rank {rank}', flush=True)
                input_tensor, fwd_wait_handles = \
                    p2p_communication.send_forward_recv_forward(
                        output_tensor, recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers,
                        overlap_p2p_comm=True)
                print(f'rank {rank} | (2) output_tensor: {output_tensor}', flush=True)
                print(f'rank {rank} | (2) input_tensor: {input_tensor}', flush=True)

                # check if last microbatch
                if k == (num_warmup_microbatches - 1) and not forward_only and \
                        not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        batch_p2p_comm=batch_p2p_comm,
                        dtype=dtype,
                        timers=timers,
                        overlap_p2p_comm=True)

                    output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
                input_tensors[next_forward_model_chunk_id].append(input_tensor)

            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

        print(f'rank {rank} | run 1F1B in steady state', flush=True)
        # Run 1F1B in steady state.
        for k in range(num_microbatches_remaining):
            print(f'----rank {rank} | beginning of 1F1B iter {k}/{num_microbatches_remaining}----', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_first_stage(ignore_virtual=True): {parallel_state.is_pipeline_first_stage(ignore_virtual=True)}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_first_stage(): {parallel_state.is_pipeline_first_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_last_stage(ignore_virtual=True): {parallel_state.is_pipeline_last_stage(ignore_virtual=True)}', flush=True)
            print(f'rank {rank} | parallel_state.is_pipeline_last_stage(): {parallel_state.is_pipeline_last_stage()}', flush=True)
            print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_rank(): {parallel_state.get_virtual_pipeline_model_parallel_rank()}', flush=True)
            print(f'rank {rank} | parallel_state.get_virtual_pipeline_model_parallel_world_size(): {parallel_state.get_virtual_pipeline_model_parallel_world_size()}', flush=True)

            # Forward pass.
            forward_k = k + num_warmup_microbatches

            if overlap_p2p_comm:
                if fwd_wait_handles is not None:
                    for req in fwd_wait_handles:
                        req.wait()

                deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

                output_tensor = forward_step_helper(forward_k)

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

                # Last virtual stage no activation tensor to send
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_model_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(
                        forward_k - (pipeline_model_parallel_size - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                                    forward=True)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                input_tensor, fwd_wait_handles = \
                    p2p_communication.send_forward_recv_forward(
                        output_tensor, recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers,
                        overlap_p2p_comm=True)
                # assert fwd_wait_handles is not None

                if bwd_wait_handles is not None:
                    for req in bwd_wait_handles:
                        req.wait()

                # Backward pass.
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

                # First virtual stage no activation gradient tensor to send
                if parallel_state.is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if the current virtual stage has an activation gradient tensor to receive
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_model_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (pipeline_model_parallel_size - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k + 1, forward=False
                    )

                output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                    overlap_p2p_comm=True)

            else: # no p2p overlap
                output_tensor = forward_step_helper(forward_k)

                # Backward pass.
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                # Send output_tensor and input_tensor_grad, receive input_tensor
                # and output_tensor_grad.

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
                if parallel_state.is_pipeline_last_stage():
                    output_tensor = None

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
                if parallel_state.is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_model_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(
                        forward_k - (pipeline_model_parallel_size - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                                    forward=True)

                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_model_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (pipeline_model_parallel_size - 1), forward=False)
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                                    forward=False)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Communicate tensors.
                input_tensor, output_tensor_grad = \
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers)
                deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(
                    output_tensor_grad)

        deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

        # Run cooldown backward passes (flush out pipeline).
        if not forward_only:
            if overlap_p2p_comm and bwd_wait_handles is not None:
                for wait_handle in bwd_wait_handles:
                    wait_handle.wait()

            if all_warmup_microbatches:
                output_tensor_grads[num_model_chunks-1].append(
                    p2p_communication.recv_backward(tensor_shape,
                                                    dtype=dtype,
                                                    batch_p2p_comm=batch_p2p_comm,
                                                    timers=timers))
            for k in range(num_microbatches_remaining, total_num_microbatches):
                input_tensor_grad = backward_step_helper(k)
                next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (num_model_chunks - 1):
                        recv_next = False
                if k == (total_num_microbatches - 1):
                    recv_next = False
                output_tensor_grads[next_backward_model_chunk_id].append(
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers))

        # Launch any remaining grad reductions
        enable_grad_sync()
        if grad_sync_func is not None:
            params = []
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    params.extend(model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
            if params:
                grad_sync_func(params)

        print(f'################## rank {rank} | leaving forward_backward_pipelining_with_interleaving ##################', flush=True)
        return forward_data_store

# TODO (gkroiz): update this for non-uniformness
def get_tensor_shapes(*,
                      pipeline_rank: int,
                      dist_rank: int,
                      model_type: ModelType,
                      tensor_shape: Shape,
                      decoder_seq_length: int,
                      sequence_parallel: bool):
    # Determine right tensor sizes (based on position of pipeline_rank with respect to split
    # pipeline_rank) and model size.
    # Send two tensors if model is T5 and pipeline_rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and pipeline_rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    # micro_batch_size will be None when using LayerUnitTestStrategy and needs to be defined
    seq_length, micro_batch_size, hidden_size = tensor_shape
    print(f'rank {dist_rank} | micro_batch_size before: ', micro_batch_size)
    
    micro_batch_size = parallel_state.get_ranks_micro_batch_size(dist_rank)
    print(f'rank {dist_rank} | micro_batch_size after: ', micro_batch_size)

    if sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()

    if model_type == ModelType.encoder_and_decoder:
        if sequence_parallel:
            decoder_seq_length = decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()

        if parallel_state.is_pipeline_stage_before_split(pipeline_rank):
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    return tensor_shapes



def recv_forward(tensor_shapes, dtype, timers):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, dtype,
                                                                timers=timers))
    return input_tensors


def recv_backward(tensor_shapes, dtype, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, dtype,
                                                                       timers=timers))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, timers=timers)


def send_backward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, timers=timers)


def send_forward_recv_backward(output_tensors, tensor_shapes, dtype, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, dtype, timers=timers)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, dtype, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, dtype, timers=timers)
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(*,
                                                     forward_step_func,
                                                     data_iterator: Union[Iterator, List[Iterator]],
                                                     model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                     num_microbatches: int,
                                                     dtype: torch.dtype,
                                                     tensor_shape: Shape,
                                                     decoder_seq_length: Optional[int] = None,
                                                     grad_scaler: Callable = None,
                                                     sequence_parallel: bool = False,
                                                     overlap_p2p_comm: bool = False,
                                                     batch_p2p_comm: bool = True,
                                                     forward_only: bool = False,
                                                     timers: Callable = None,
                                                     collect_non_loss_data: bool = False,
                                                     enable_autocast: bool = False,
                                                     deallocate_pipeline_outputs: bool = False,
                                                     no_sync_func: Optional[Callable] = None,
                                                     grad_sync_func: Optional[Callable] = None,
                                                     param_sync_func: Optional[Callable] = None, # unused
                                                     ):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if overlap_p2p_comm:
        raise ValueError("Non-interleaved pipeline parallelism does not support overlapping p2p communication")

    if not batch_p2p_comm:
        raise ValueError("Non-interleaved pipeline parallelism only supports using batched p2p communication")

    # Disable async grad reductions
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    model_type = get_model_type(model)

    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    rank = torch.distributed.get_rank()

    print('calling get_tensor_shapes', flush=True)
    recv_tensor_shapes = get_tensor_shapes(pipeline_rank=pipeline_rank-1,
                                           dist_rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(pipeline_rank=pipeline_rank,
                                           dist_rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []


    print(f'rank {rank} BEGIN 1F1B PIPELINE', flush=True)
    print(f'rank {rank} | recv_tensor_shapes: {recv_tensor_shapes}', flush=True)
    print(f'rank {rank} | send_tensor_shapes: {send_tensor_shapes}', flush=True)

    # Run warmup forward passes.
    warmup_rng = nvtx.start_range(message="fwd_pass_warmup", color="green")
    print(f'rank {rank} | ', flush=True)
    print(f'rank {rank} | num_warmup_microbatches: {num_warmup_microbatches}', flush=True)
    for i in range(num_warmup_microbatches):
        if i % 2 == 0:
            nvtx.mark(message=f"fwd_pass_warmup: {i}, rank: {rank}", color="yellow")
        else:
            nvtx.mark(message=f"fwd_pass_warmup: {i}, rank: {rank}", color="orange")
        print(f'************ rank {rank} recv_foward ************', flush=True)
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)
        print(f'---rank {rank} | recv_foward OUTPUT_TENSOR: {input_tensor}', flush=True)
        print('***************************************', flush=True)
        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        print(f'************ rank {rank} warmup fwd_step output ************', flush=True)
        print(f'---rank {rank} | forward_step OUTPUT TENSOR: {output_tensor}', flush=True)
        print(f'************ rank {rank} send fwd ************', flush=True)
        send_forward(output_tensor, send_tensor_shapes, timers=timers)
        print(f'---rank {rank} | send_foward OUTPUT TENSOR: {output_tensor}', flush=True)
        print('***************************************', flush=True)
        
        print(f'rank {rank} | not forward_only: {not forward_only}', flush=True)
        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)

    nvtx.end_range(warmup_rng)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    print(f'rank {rank} | num_microbatches_remaining {num_microbatches_remaining}', flush=True)
    if num_microbatches_remaining > 0:
        print(f'************ rank {rank} recv_foward before 1F1B ************', flush=True)
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)
        print(f'---rank {rank} | recv_forward OUTPUT TENSOR: {input_tensor}', flush=True)
        print('***************************************', flush=True)

    steady_state_rng = nvtx.start_range(message="fwd_pass_steady_state", color="darkgreen")
    print(f'************ rank {rank} in 1F1B ************', flush=True)
    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        print(f'************ rank {rank} FORWARD STEP ************', flush=True)
        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        print(f'************ rank {rank} output_tensor ************', flush=True)
        print(f'---rank {rank} | forward_step OUTPUT TENSOR: {output_tensor}', flush=True)
        print('***************************************', flush=True)

        print(f'rank {rank} | forward_only: {forward_only}', flush=True)
        if forward_only:
            print(f'************ rank {rank} send fwd ************', flush=True)
            send_forward(output_tensor, send_tensor_shapes, timers=timers)
            print(f'---rank {rank} | send_forward OUTPUT TENSOR: {output_tensor}', flush=True)
            print('***************************************', flush=True)

            print(f'rank {rank} | not last_iteration: {not last_iteration}', flush=True)
            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)
                print(f'************ rank {rank} recv_foward ************', flush=True)
                print(f'---rank {rank} | recv_forward OUTPUT TENSOR: {input_tensor}', flush=True)
                print('***************************************', flush=True)

        else:
            print(f'************ rank {rank} send_forward_recv_backward ************', flush=True)
            output_tensor_grad = \
                send_forward_recv_backward(output_tensor,
                                           send_tensor_shapes, dtype,
                                           timers=timers)
            print(f'--rank {rank} | send_forward_recv_backward INPUT TENSOR: {output_tensor}', flush=True)
            print(f'--rank {rank} | send_forward_recv_backward OUTPUT TENSOR: {output_tensor_grad}', flush=True)
            print('***************************************', flush=True)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            print(f'************ rank {rank} BACKWARD STEP ************', flush=True)
            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
            print(f'---rank {rank} | backward_step OUTPUT TENSOR: {input_tensor_grad}', flush=True)
            print('***************************************', flush=True)

            print(f'rank {rank} | is last_iteration: {last_iteration}', flush=True)
            if last_iteration:
                input_tensor = None
                print(f'************ rank {rank} send_backward ************', flush=True)
                send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
                print(f'---rank {rank} | send_backward INPUT TENSOR: {input_tensor_grad}', flush=True)
                print('***************************************', flush=True)
            else:
                print(f'************ rank {rank} send_backward_recv_forward ************', flush=True)
                input_tensor = \
                    send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, dtype, timers=timers)
                print(f'---rank {rank} | send_backward_recv_forward INPUT TENSOR: {input_tensor_grad}', flush=True)
                print(f'---rank {rank} | send_backward_recv_forward OUTPUT TENSOR: {input_tensor}', flush=True)
                print('***************************************', flush=True)

    nvtx.end_range(steady_state_rng)
    # Run cooldown backward passes.
    print(f'rank {rank} | cooldown backward passes', flush=True)
    if not forward_only:
        print(f'rank {rank} | num_warmup_microbatches: {num_warmup_microbatches}', flush=True)
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches-1:
                if grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, dtype, timers=timers)
            print(f'************ rank {rank} recv_backward ************', flush=True)
            print(f'rank {rank} | recv_backward OUTPUT TENSOR: {output_tensor_grad}', flush=True)
            print('***************************************', flush=True)
            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
            print(f'************ rank {rank} backward_step ************', flush=True)
            print(f'rank {rank} | backward_step OUTPUT TENSOR: {input_tensor_grad}', flush=True)
            print('***************************************', flush=True)

            print(f'************ rank {rank} send_backward ************', flush=True)
            send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
            print(f'---rank {rank} | send_backward INPUT TENSOR: {input_tensor_grad}', flush=True)
            print('***************************************', flush=True)
    # Launch any remaining grad reductions
    if no_sync_context is not None:
        enable_grad_sync()
        if grad_sync_func is not None:
            grad_sync_func(model.parameters())

    print(f'rank {rank} END 1F1B PIPELINE', flush=True)
    return forward_data_store
