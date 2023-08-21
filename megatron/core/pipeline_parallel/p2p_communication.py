# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from functools import reduce
import operator
from typing import Optional, List, Union, Callable, Tuple

import torch
import nvtx

from megatron import core
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_groups,
    get_pipeline_model_parallel_prev_ranks,
    get_pipeline_model_parallel_next_ranks,
    get_using_layer_unit_test_strategy,
    get_ranks_micro_batch_size
)

# Types
Shape = Union[List[int], torch.Size]

def _communicate_shapes(tensor_send_next, tensor_send_prev,
                        recv_prev, recv_next,
                        use_ring_exchange_p2p):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    if recv_next:
        recv_next_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(tensor_send_prev.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(tensor_send_next.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)

    if use_ring_exchange_p2p:
        assert get_using_layer_unit_test_strategy() == False, \
            'use_ring_exchange_p2p not implemented for LayerUnitTestStrategy'
        torch.distributed.ring_exchange(tensor_send_prev=send_prev_shape_tensor,
                                        tensor_recv_prev=recv_prev_shape_tensor,
                                        tensor_send_next=send_next_shape_tensor,
                                        tensor_recv_next=recv_next_shape_tensor,
                                        group=get_pipeline_model_parallel_group())
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_ranks())
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_ranks())
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, send_next_shape_tensor,
                get_pipeline_model_parallel_next_ranks())
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_next_shape_tensor,
                get_pipeline_model_parallel_next_ranks())
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape

def _batched_p2p_ops(*,
                     tensor_send_prev: Optional[torch.Tensor],
                     tensor_recv_prev: Optional[torch.Tensor],
                     tensor_send_next: Optional[torch.Tensor],
                     tensor_recv_next: Optional[torch.Tensor],
                     group: torch.distributed.ProcessGroup):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev,
            get_pipeline_model_parallel_prev_ranks(),
            group)
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev,
            get_pipeline_model_parallel_prev_ranks(),
            group)
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next,
            get_pipeline_model_parallel_next_ranks(),
            group)
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next,
            get_pipeline_model_parallel_next_ranks(),
            group)
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs

def _batched_p2p_ops_w_components(*,
                     tensor_send_prev: Optional[torch.Tensor],
                     tensors_recv_prev: Optional[torch.Tensor],
                     tensor_send_next: Optional[torch.Tensor],
                     tensors_recv_next: Optional[torch.Tensor],
                     groups: List[torch.distributed.ProcessGroup]):
    ops = []
    if tensor_send_prev is not None:
        num_groups = len(groups)
        # if fan-in, num_groups = 1
        if num_groups == 1:
            split_tensors = [tensor_send_prev]
        # if fan-out, num_groups > 1
        elif num_groups > 1:
            split_sections = [
                get_ranks_micro_batch_size(prev_node) for prev_node in get_pipeline_model_parallel_prev_ranks()[i]
            ]
            split_tensors = torch.split(tensor_send_prev, split_sections, dim=1)
        else:
            raise ValueError("This error message should not appear, please file a bug.")
        for i, group in enumerate(groups):
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, split_tensors[i],
                get_pipeline_model_parallel_prev_ranks()[i],
                group)
            ops.append(send_prev_op)
    if tensors_recv_prev is not None:
        for i, _ in enumerate(groups):
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensors_recv_prev[i],
                get_pipeline_model_parallel_prev_ranks()[i],
                groups[i])
            ops.append(recv_prev_op)
    if tensor_send_next is not None:
        num_groups = len(groups)
        # if fan-in, num_groups = 1
        if num_groups == 1:
            split_tensors = [tensor_send_next]
        # if fan-out, num_groups > 1
        elif num_groups >= 1:
            split_sections = [
                get_ranks_micro_batch_size(next_node) for next_node in get_pipeline_model_parallel_next_ranks()[i]
            ]
            split_tensors = torch.split(tensor_send_next, split_sections, dim=1)
        else:
            raise ValueError("This error message should not appear, please file a bug.")
        for i, group in enumerate(groups):
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, split_tensors[i],
                get_pipeline_model_parallel_next_ranks()[i],
                group)
            ops.append(send_next_op)
    if tensors_recv_next is not None:
        for i, _ in enumerate(groups):
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensors_recv_next[i],
                get_pipeline_model_parallel_next_ranks()[i],
                groups[i])
            ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs

def _p2p_ops(*,
             tensor_send_prev: Optional[torch.Tensor],
             tensor_recv_prev: Optional[torch.Tensor],
             tensor_send_next: Optional[torch.Tensor],
             tensor_recv_next: Optional[torch.Tensor],
             group: torch.distributed.ProcessGroup):
    reqs = []
    if tensor_send_next is not None:
        send_next_req = torch.distributed.isend(
            tensor=tensor_send_next,
            dst=get_pipeline_model_parallel_next_ranks(),
            group=group,
        )
        reqs.append(send_next_req)

    if tensor_recv_prev is not None:
        recv_prev_req = torch.distributed.irecv(
            tensor=tensor_recv_prev,
            src=get_pipeline_model_parallel_prev_ranks(),
            group=group,
        )
        reqs.append(recv_prev_req)

    if tensor_send_prev is not None:
        send_prev_req = torch.distributed.isend(
            tensor=tensor_send_prev,
            dst=get_pipeline_model_parallel_prev_ranks(),
            group=group,
        )
        reqs.append(send_prev_req)

    if tensor_recv_next is not None:
        recv_next_req = torch.distributed.irecv(
            tensor=tensor_recv_next,
            src=get_pipeline_model_parallel_next_ranks(),
            group=group,
        )
        reqs.append(recv_next_req)
    return reqs

def _p2p_ops_w_components(*,
             tensor_send_prev: Optional[torch.Tensor],
             tensors_recv_prev: Optional[torch.Tensor],
             tensor_send_next: Optional[torch.Tensor],
             tensors_recv_next: Optional[torch.Tensor],
             groups: List[torch.distributed.ProcessGroup] = None):
    reqs = []
    if tensor_send_next is not None:
        num_groups = len(groups)
        # if fan-in (from current rank), num_groups = 1
        if num_groups == 1:
            split_tensors = [tensor_send_next]
        # if fan-out (from current rank), num_groups > 1
        else:
            split_tensors = tensor_send_next.chunk(num_groups)
        for i, _ in enumerate(groups):
            send_next_req = torch.distributed.isend(
                tensor=split_tensors[i],
                dst=get_pipeline_model_parallel_next_ranks()[i],
                group=groups[i],
            )
            reqs.append(send_next_req)

    if tensors_recv_prev is not None:
        for i, _ in enumerate(groups):
            recv_prev_req = torch.distributed.irecv(
                tensor=tensors_recv_prev[i],
                src=get_pipeline_model_parallel_prev_ranks()[i],
                group=groups[i],
            )
            reqs.append(recv_prev_req)

    if tensor_send_prev is not None:
        num_groups = len(groups)
        # if fan-in (from current rank), num_groups = 1
        # if fan-out (from current rank), num_groups > 1
        split_tensors = tensor_send_prev.chunk(num_groups)
        for i in range(num_groups):
            send_prev_req = torch.distributed.isend(
                tensor=split_tensors[i],
                dst=get_pipeline_model_parallel_prev_ranks()[i],
                group=groups[i],
            )
            reqs.append(send_prev_req)

    if tensors_recv_next is not None:
        for i, _ in enumerate(groups):
            recv_next_req = torch.distributed.irecv(
                tensor=tensors_recv_next[i],
                src=get_pipeline_model_parallel_next_ranks()[i],
                group=groups[i],
            )
            reqs.append(recv_next_req)


    return reqs

def _communicate(*, tensor_send_next: Optional[torch.Tensor],
                 tensor_send_prev: Optional[torch.Tensor],
                 recv_prev: bool,
                 recv_next: bool,
                 tensor_shape: Shape,
                 batch_p2p_comm: bool = True,
                 wait_on_reqs: bool = True,
                 dtype: Optional[torch.dtype],
                 variable_seq_lengths: bool = False,
                 use_ring_exchange_p2p: bool = False,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        batch_p2p_comm (boolean, required):
            If true use batch_isend_irecv, otherwise use individual
            isend and irecv calls.

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

        dtype (torch.dtype, required if either recv_{prev,next} is True):
            this must be the type of the tensors that will be
            received, will typically be params_dtype, but in the case
            of fp32 residual connections might be torch.float.

        variable_seq_lengths (bool, optional, default=False):
            Support for variable sequence lengths across
            microbatches. Setting this communicates the size of
            tensors during pipeline parallelism communication, because
            of this extra overhead it should only be set if the
            sequence length is not constant during training.

        use_ring_exchange_p2p (bool, optional, default = False):
            Use custom ring_exchange kernel instead of
            torch.distributed.batch_isend_irecv(). Requires custom
            built torch with torch.distributed.ring_exchange.


    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # This will come from config in the next version, for now hard
    # code it here to match existing functionality.
    batch_p2p_sync = True

    if not variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = \
            _communicate_shapes(tensor_send_next,
                                tensor_send_prev,
                                recv_prev,
                                recv_next)

    if recv_prev:
        if dtype is None:
            raise RuntimeError("dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(recv_prev_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_next:
        if dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(recv_next_shape,
                                       requires_grad=True,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Send tensors in both the forward and backward directions as appropriate.
    if use_ring_exchange_p2p:
        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []
        p2p_func = _ring_exchange_wrapper
    elif batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops_w_components if get_using_layer_unit_test_strategy() else _batched_p2p_ops
    else:
        p2p_func = _p2p_ops_w_components if get_using_layer_unit_test_strategy() else _p2p_ops

    if get_using_layer_unit_test_strategy():
        tensors_recv_prev, tensors_recv_next = None, None
        if (tensor_recv_prev is not None):
            prev_ranks = get_pipeline_model_parallel_prev_ranks()
            # if fan-in (from current rank), len(tensors_recv_prev) = 1
            if len(prev_ranks) == 1:
                split_tensors = [tensor_recv_prev]
            # if fan-out (from current rank), len(tensors_recv_prev) > 1
            elif len(prev_ranks) > 1:
                split_sections = [
                    get_ranks_micro_batch_size(prev_node) for prev_node in get_pipeline_model_parallel_prev_ranks()[i]
                ]
                # check that the split sections add up to the micro_batch_size
                assert sum(split_sections) == tensor_recv_prev.size()[1]
                split_tensors = torch.split(tensor_recv_prev, split_sections, dim=1)
            tensors_recv_prev = []
            for i, _ in enumerate(prev_ranks):
                tensors_recv_prev.append(split_tensors[i])
        if (tensor_recv_next is not None):
            next_ranks = get_pipeline_model_parallel_next_ranks()
            # if fan-in (from current rank), len(tensors_recv_next) = 1
            if len(next_ranks) == 1:
                split_tensors = [tensor_recv_next]
            # if fan-out (from current rank), len(tensors_recv_next) > 1
            elif len(next_ranks) > 1:
                split_sections = [
                    get_ranks_micro_batch_size(next_node) for next_node in get_pipeline_model_parallel_next_ranks()[i]
                ]
                # check that the split sections add up to the micro_batch_size
                assert sum(split_sections) == tensor_recv_prev.size()[1]
                split_tensors = torch.split(tensor_recv_next, split_sections, dim=1)
            tensors_recv_next = []
            for i, _ in enumerate(next_ranks):
                tensors_recv_next.append(split_tensors[i])

        # need to define all tensors here

        # if there are multiple prev_groups, fan in, tensor size should remain the same
        # if there are multiple next_groups, fan out, tensor size should be split
        reqs = p2p_func(tensor_send_prev=tensor_send_prev,
                        tensors_recv_prev=tensors_recv_prev,
                        tensor_send_next=tensor_send_next,
                        tensors_recv_next=tensors_recv_next,
                        groups=get_pipeline_model_parallel_groups())
    else:
        reqs = p2p_func(tensor_send_prev=tensor_send_prev,
                        tensor_recv_prev=tensor_recv_prev,
                        tensor_send_next=tensor_send_next,
                        tensor_recv_next=tensor_recv_next,
                        group=get_pipeline_model_parallel_group())

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if batch_p2p_comm and batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    # TODO: combine tensors for fan-in
    if get_using_layer_unit_test_strategy():
        if tensors_recv_prev is not None:
            tensor_recv_prev = torch.cat(tensors_recv_prev)
        if tensors_recv_next is not None:
            tensor_recv_next = torch.cat(tensors_recv_next)

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward(tensor_shape: Shape,
                 dtype: torch.dtype,
                 batch_p2p_comm: bool = True,
                 timers: Callable = None) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).


    See _communicate for argument details.
    """

    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=dtype)
        if timers is not None:
            timers('forward-recv').stop()
    return input_tensor


def recv_backward(tensor_shape: Shape,
                  dtype: torch.dtype,
                  batch_p2p_comm: bool = True,
                  timers: Callable = None) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=dtype)
        if timers is not None:
            timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor: torch.Tensor,
                 batch_p2p_comm: bool = True,
                 timers: Callable = None) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not core.parallel_state.is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send', log_level=2).start()
        rng = nvtx.start_range(message="send_forward", color="orange")
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            batch_p2p_comm=batch_p2p_comm,
            dtype=None)
        nvtx.end_range(rng)
        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad: torch.Tensor,
                  batch_p2p_comm: bool = True,
                  timers: Callable = None) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not core.parallel_state.is_pipeline_first_stage():
        if timers is not None:
            timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            batch_p2p_comm=batch_p2p_comm,
            dtype=None)
        if timers is not None:
            timers('backward-send').stop()


def send_forward_recv_backward(output_tensor: torch.Tensor,
                               tensor_shape: Shape,
                               dtype: torch.dtype,
                               batch_p2p_comm: bool = True,
                               timers: Callable = None) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad,_ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=dtype)
        if timers is not None:
            timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad: torch.Tensor,
                               tensor_shape: Shape,
                               dtype: torch.dtype,
                               batch_p2p_comm: bool = True,
                               timers: Callable = None) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=dtype)
        if timers is not None:
            timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor: torch.Tensor,
                              recv_prev: bool,
                              tensor_shape: Shape,
                              dtype: torch.dtype,
                              batch_p2p_comm: bool = True,
                              overlap_p2p_comm: bool = False,
                              timers: Callable = None) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if timers is not None:
        timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        batch_p2p_comm=batch_p2p_comm,
        wait_on_reqs=(not overlap_p2p_comm),
        dtype=dtype)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_backward_recv_backward(input_tensor_grad: torch.Tensor,
                                recv_next: bool,
                                tensor_shape: Shape,
                                dtype: torch.dtype,
                                batch_p2p_comm: bool = True,
                                overlap_p2p_comm: bool = False,
                                timers: Callable = None) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if timers is not None:
        timers('backward-send-backward-recv', log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        batch_p2p_comm=batch_p2p_comm,
        wait_on_reqs=(not overlap_p2p_comm),
        dtype=dtype)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        dtype: torch.dtype,
        batch_p2p_comm: bool = True,
        timers: Callable = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv',
               log_level=2).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        batch_p2p_comm=batch_p2p_comm,
        dtype=dtype)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
