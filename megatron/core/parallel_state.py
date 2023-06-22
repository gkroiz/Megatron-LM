# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import torch
from typing import Optional

from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model component parallel group that the current rank belongs to.
_PIPELINE_COMPONENT_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# FP8 amax reduction group.
_AMAX_REDUCTION_GROUP = None
# Previous component pipeline group that the current rank belongs to.
_PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP = None
# Next component pipeline group that the current rank belongs to.
_NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP = None

# Previous pipeline model rank based on current rank.
_PREV_PIPELINE_MODEL_PARALLEL_RANK = None
# Next pipeline model rank based on current rank.
_NEXT_PIPELINE_MODEL_PARALLEL_RANK = None
# First pipeline model rank based on current rank.
_FIRST_PIPELINE_MODEL_PARALLEL_RANK = None
# Last pipeline model rank based on current rank.
_LAST_PIPELINE_MODEL_PARALLEL_RANK = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_COMPONENT_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# number of layers in the component that the current rank belongs to
_NUM_COMPONENT_LAYERS = None

# TODO (gersonkroiz) Verify this
def initialize_model_components_parallel(
    parallelization_specs: dict,
    use_fp8: bool = False,
):
    """Initialize data parallel groups for component of the model.
    
    Arguments:
        parallelization_specs: (dict, required)
            contains specifications for initializing the 
            component parallel groups
    """
    assert torch.distributed.is_initialized()
    world_sizes = {}
    tensor_model_parallel_group_sizes = {}
    data_parallel_group_sizes = {}
    pipeline_model_parallel_group_sizes = {}

    all_num_tensor_model_parallel_groups = {}
    all_num_pipeline_model_parallel_groups = {}
    all_num_data_parallel_groups = {}
    
    all_data_parallel_group_ranks = {}
    all_gpu_ranks = {}

    for k in parallelization_specs:
        world_sizes[k] = len(parallelization_specs[k]["gpu_ranks"])
        tensor_model_parallel_group_sizes[k] = parallelization_specs[k]["tensor_model_parallel_group_size"]
        data_parallel_group_sizes[k] = parallelization_specs[k]["data_parallel_group_size"]
        pipeline_model_parallel_group_sizes[k] = parallelization_specs[k]["pipeline_model_parallel_group_size"]
        all_num_tensor_model_parallel_groups[k] = world_sizes[k] // tensor_model_parallel_group_sizes[k]
        all_num_pipeline_model_parallel_groups[k] = world_sizes[k] // pipeline_model_parallel_group_sizes[k]
        all_num_data_parallel_groups[k] = world_sizes[k] // data_parallel_group_sizes[k]
        
        all_data_parallel_group_ranks[k] = []
        all_gpu_ranks[k] = parallelization_specs[k]['gpu_ranks']
        
    for k in parallelization_specs:
        if world_sizes[k] % (tensor_model_parallel_group_sizes[k] * pipeline_model_parallel_group_sizes[k]) != 0:
            raise RuntimeError(
                f"component world_size ({world_size[k]}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_group_sizes[k]}) x pipeline_model_parallel_size ({pipeline_model_parallel_group_sizes[k]})"
            )

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    
    for k in parallelization_specs:
        for i in range(pipeline_model_parallel_group_sizes[k]):
            start_rank = i * all_num_pipeline_model_parallel_groups[k]
            end_rank = (i + 1) * all_num_pipeline_model_parallel_groups[k]
            for j in range(tensor_model_parallel_group_sizes[k]):
                ranks = range(all_gpu_ranks[k][start_rank + j], all_gpu_ranks[k][end_rank-1]+1, tensor_model_parallel_group_sizes[k])
                all_data_parallel_group_ranks[k].append(list(ranks))
                group = torch.distributed.new_group(ranks)
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    # TODO: update this
                    _DATA_PARALLEL_GROUP = group
                    _DATA_PARALLEL_GROUP_GLOO = group_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for k in parallelization_specs:
        for i in range(data_parallel_group_sizes[k]):
            ranks = [data_parallel_group_ranks[i]
                    for data_parallel_group_ranks in all_data_parallel_group_ranks[k]]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for k in parallelization_specs:
        for i in range(all_num_tensor_model_parallel_groups[k]):
            ranks = range(all_gpu_ranks[k][i * tensor_model_parallel_group_sizes[k]],
                          all_gpu_ranks[k][((i + 1) * tensor_model_parallel_group_sizes[k])-1]+1)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel and component pipeline connector groups
    global _PIPELINE_COMPONENT_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _PREV_PIPELINE_MODEL_PARALLEL_RANK
    global _NEXT_PIPELINE_MODEL_PARALLEL_RANK
    global _FIRST_PIPELINE_MODEL_PARALLEL_RANK
    global _LAST_PIPELINE_MODEL_PARALLEL_RANK
    global _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP
    global _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP

    first_component_name = list(parallelization_specs.keys())[0] 
    last_component_name = list(parallelization_specs.keys())[-1]

    # arrays to define full pipeline model parallel groups
    full_pipeline_model_parallel_groups = [[] for _ in range(all_num_pipeline_model_parallel_groups[first_component_name])]
    full_pipeline_model_parallel_groups_ranks_tracker = [_ for _ in range(all_num_pipeline_model_parallel_groups[first_component_name])]

    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    for k in parallelization_specs:
        for i in range(all_num_pipeline_model_parallel_groups[k]):
            ranks = range(
                all_gpu_ranks[k][i],
                all_gpu_ranks[k][-1]+1,
                all_num_pipeline_model_parallel_groups[k]
            )
            group = torch.distributed.new_group(ranks)

            assert full_pipeline_model_parallel_groups_ranks_tracker[i] in ranks
            for tmp_rank in ranks:
                full_pipeline_model_parallel_groups[i].append(tmp_rank)
                full_pipeline_model_parallel_groups_ranks_tracker[i] = tmp_rank
            full_pipeline_model_parallel_groups_ranks_tracker[i] += all_num_pipeline_model_parallel_groups[k]

            # define global vars for full model pipeline parallel group
            if rank in ranks:

                # define first and last ranks in full model pipeline parallel group
                _FIRST_PIPELINE_MODEL_PARALLEL_RANK = all_gpu_ranks[first_component_name][i]

                # TODO: write simpler logic
                last_component_pipeline_model_parallel_ranks = range(
                    all_gpu_ranks[last_component_name][i],
                    all_gpu_ranks[last_component_name][world_sizes[last_component_name]-1]+1,
                    all_num_pipeline_model_parallel_groups[last_component_name]
                )
                _LAST_PIPELINE_MODEL_PARALLEL_RANK = last_component_pipeline_model_parallel_ranks[-1]

                _PIPELINE_COMPONENT_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks

                # define previous and next ranks in the full model pipeline parallel group
                if not (rank == ranks[0] and k == list(parallelization_specs.keys())[0]):
                    _PREV_PIPELINE_MODEL_PARALLEL_RANK = rank - all_num_pipeline_model_parallel_groups[k]
                else:
                    _PREV_PIPELINE_MODEL_PARALLEL_RANK = -1

                if not (rank == ranks[-1] and k == list(parallelization_specs.keys())[-1]):
                    _NEXT_PIPELINE_MODEL_PARALLEL_RANK = rank + all_num_pipeline_model_parallel_groups[k]
                else:
                    _NEXT_PIPELINE_MODEL_PARALLEL_RANK = -1

            # define component pipeline connector groups
            # TODO: need to modify for fan-in/out
            if k != list(parallelization_specs.keys())[-1]:
                # establish the groups connecting this component with next component
                connector_ranks = [ranks[-1], (ranks[-1] + all_num_pipeline_model_parallel_groups[k]) % sum(world_sizes.values())]
                connector_group = torch.distributed.new_group(connector_ranks)

                if rank in connector_ranks:
                    if rank == connector_ranks[0]:
                        assert _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP is None
                        _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP = connector_group

                    if rank == connector_ranks[1]:
                        assert _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP is None
                        _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP = connector_group


    for grouping in full_pipeline_model_parallel_groups:
        group = torch.distributed.new_group(grouping) 
        if rank in grouping:
            _PIPELINE_MODEL_PARALLEL_GROUP = group

    # Build the embedding groups (first and last rank in each pipeline model-parallel group).
    # The embedding groups are defined based on the entire model, not individual components

    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        'position embedding group is already initialized'
    for i in range(all_num_pipeline_model_parallel_groups[first_component_name]):
        ranks = range(all_gpu_ranks[first_component_name][i], 
                      all_gpu_ranks[last_component_name][world_sizes[last_component_name]-1]+1, 
                      all_num_pipeline_model_parallel_groups[first_component_name])
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the FP8 groups.
    global _AMAX_REDUCTION_GROUP
    assert _AMAX_REDUCTION_GROUP is None, \
        'FP8 amax reduction group is already initialized'
    if use_fp8:
        for k in parallelization_specs:
            amax_group_size: int = tensor_model_parallel_group_sizes[k] * data_parallel_group_sizes[k]
            num_amax_groups: int = world_sizes[k] // amax_group_size
            for i in range(num_amax_groups):
                start_rank = i * amax_group_size
                end_rank = (i + 1) * amax_group_size
                ranks = range(all_gpu_ranks[k][start_rank], all_gpu_ranks[k][end_rank-1]+1)
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _AMAX_REDUCTION_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()
    

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_fp8: bool = False,
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_fp8 (bool, default = False):
            Construct GPU groups needed for FP8 training, namely for
            amax reduction across the product of the data-parallel and
            tensor-parallel groups.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    data_parallel_size: int = world_size // (tensor_model_parallel_size *
                                             pipeline_model_parallel_size)

    num_tensor_model_parallel_groups: int  = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError("pipeline-model-parallel size should be greater than 2 with "
                               "interleaved schedule")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_COMPONENT_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_COMPONENT_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank],
                                       ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the FP8 groups.
    global _AMAX_REDUCTION_GROUP
    assert _AMAX_REDUCTION_GROUP is None, \
        'FP8 amax reduction group is already initialized'
    if use_fp8:
        amax_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_amax_groups: int = world_size // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _AMAX_REDUCTION_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_COMPONENT_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_component_parallel_group(direction: Optional[str] = ''):
    """Get the pipeline component parallel group the caller rank belongs to."""
    """If direction is specific, get the next/prev component parallel connector group"""
    if direction != '':
        assert direction in ['next', 'prev']
        if direction == 'next' and is_pipeline_component_last_stage():
            assert _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP is not None
            return _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP
        if direction == 'prev' and is_pipeline_component_first_stage():
            assert _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP is not None
            return _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP
    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_COMPONENT_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo():
    """Get the data parallel group-gloo the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP_GLOO is not None, \
        'data parallel group-gloo is not initialized'
    return _DATA_PARALLEL_GROUP_GLOO


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, \
        'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group():
    """Get the FP8 amax reduction group the caller rank belongs to."""
    assert _AMAX_REDUCTION_GROUP is not None, \
        'FP8 amax reduction group is not initialized'
    return _AMAX_REDUCTION_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_component_parallel_world_size(world_size):
    """Set the pipeline component parallel size"""
    global _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_pipeline_component_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_component_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_component_parallel_rank(rank):
    """Set pipeline component parallel rank."""
    global _MPU_PIPELINE_COMPONENT_PARALLEL_RANK
    _MPU_PIPELINE_COMPONENT_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_component_parallel_rank():
    """Return my rank for the pipeline component parallel group."""
    global _MPU_PIPELINE_COMPONENT_PARALLEL_RANK
    if _MPU_PIPELINE_COMPONENT_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_COMPONENT_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_component_parallel_group())


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_component_first_stage():
    """Return True if in the first pipeline component-parallel stage, False otherwise."""
    # TODO: implement virtual conditional
    return get_pipeline_component_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def is_pipeline_component_last_stage():
    """Return True if in the last pipeline component-parallel stage, False otherwise."""
    # TODO: implement virtual conditional
    return get_pipeline_component_parallel_rank() == (
        get_pipeline_component_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and \
            is_pipeline_stage_after_split(rank+1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the virtual pipeline-parallel world size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_num_component_layers():
    """Return the number of layers in component."""
    global _NUM_COMPONENT_LAYERS
    return _NUM_COMPONENT_LAYERS


def set_num_component_layers(num_layers):
    """Set the number of layers in compoment."""
    global _NUM_COMPONENT_LAYERS
    _NUM_COMPONENT_LAYERS = num_layers


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, \
        "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _FIRST_PIPELINE_MODEL_PARALLEL_RANK is not None, \
        "First pipeline parallel model rank is not initialized"
    return _FIRST_PIPELINE_MODEL_PARALLEL_RANK


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _LAST_PIPELINE_MODEL_PARALLEL_RANK is not None, \
        "Last pipeline parallel model rank is not initialized"
    return _LAST_PIPELINE_MODEL_PARALLEL_RANK


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _NEXT_PIPELINE_MODEL_PARALLEL_RANK is not None, \
        "Next pipeline parallel model rank is not initialized"
    assert _NEXT_PIPELINE_MODEL_PARALLEL_RANK != -1, \
        "Current rank does not have a a next pipeline model parallel rank"
    return _NEXT_PIPELINE_MODEL_PARALLEL_RANK


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PREV_PIPELINE_MODEL_PARALLEL_RANK is not None, \
        "Previous pipeline parallel group is not initialized"
    assert _PREV_PIPELINE_MODEL_PARALLEL_RANK != -1, \
        "Current rank does not have a a previous pipeline model parallel rank"
    return _PREV_PIPELINE_MODEL_PARALLEL_RANK


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_COMPONENT_PARALLEL_GROUP
    _PIPELINE_COMPONENT_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _AMAX_REDUCTION_GROUP
    _AMAX_REDUCTION_GROUP = None
    global _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP
    _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUP = None
    global _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP
    _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUP = None
    global _PREV_PIPELINE_MODEL_PARALLEL_RANK
    _PREV_PIPELINE_MODEL_PARALLEL_RANK = None
    global _NEXT_PIPELINE_MODEL_PARALLEL_RANK
    _NEXT_PIPELINE_MODEL_PARALLEL_RANK = None
    global _FIRST_PIPELINE_MODEL_PARALLEL_RANK
    _FIRST_PIPELINE_MODEL_PARALLEL_RANK = None
    global _LAST_PIPELINE_MODEL_PARALLEL_RANK
    _LAST_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_COMPONENT_PARALLEL_RANK
    _MPU_PIPELINE_COMPONENT_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    global _NUM_COMPONENT_LAYERS
    _NUM_COMPONENT_LAYERS = None
