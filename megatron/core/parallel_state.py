# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import torch
import math
from typing import Optional

from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = []
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = []
# Embedding group.
_EMBEDDING_GROUP = []
# Position embedding group.
_POSITION_EMBEDDING_GROUP = []
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# FP8 amax reduction group.
_AMAX_REDUCTION_GROUP = None

# Previous pipeline model rank based on current rank.
_PREV_PIPELINE_MODEL_PARALLEL_RANK = []
# Next pipeline model rank based on current rank.
_NEXT_PIPELINE_MODEL_PARALLEL_RANK = []
# First pipeline model rank based on current rank.
_FIRST_PIPELINE_MODEL_PARALLEL_RANK = []
# Last pipeline model rank based on current rank.
_LAST_PIPELINE_MODEL_PARALLEL_RANK = []

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# Mappings of all pipeline parallel groups with data_split_ratio
_FWD_MAPPINGS = {}
_BKWD_MAPPINGS = {}

# Dictionary of the micro_batch_sizes for each rank
_ALL_RANKS_MICRO_BATCH_SIZES = {}

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_COMPONENT_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_COMPONENT_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = []

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = []

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = []
_PIPELINE_COMPONENT_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# number of layers in the component that the current rank belongs to
_NUM_COMPONENT_LAYERS = None

# whether layer unit test strategy is used (boolean)
_USING_LAYER_UNIT_TEST_STRATEGY = None

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

    global _USING_LAYER_UNIT_TEST_STRATEGY
    _USING_LAYER_UNIT_TEST_STRATEGY = False

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
    assert len(_MODEL_PARALLEL_GROUP) == 0, 'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP.append(group)

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
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert len(_PIPELINE_MODEL_PARALLEL_GROUP) == 0, \
        'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert len(_EMBEDDING_GROUP) == 0, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert len(_POSITION_EMBEDDING_GROUP) == 0, \
        'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
            _PIPELINE_GLOBAL_RANKS.append(ranks)
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
            _EMBEDDING_GROUP.append(group)
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS.append(embedding_ranks)

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP.append(group)
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS.append(position_embedding_ranks)

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


def initialize_model_components_parallel(
    parallelization_specs: dict,
    micro_batch_size: int,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_fp8: bool = False,
) -> None:
    """Initialize data parallel groups for component of the model.

    Arguments:
        parallelization_specs: (dict, required)
            contains specifications for initializing the
            component parallel groups

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

    """
    assert torch.distributed.is_initialized()

    global _USING_LAYER_UNIT_TEST_STRATEGY
    _USING_LAYER_UNIT_TEST_STRATEGY = True

    world_sizes = {}
    tensor_model_parallel_group_sizes = {}
    data_parallel_group_sizes = {}
    pipeline_model_parallel_group_sizes = {}

    all_num_tensor_model_parallel_groups = {}
    all_num_pipeline_components_parallel_groups = {}
    all_num_data_parallel_groups = {}

    all_data_parallel_group_ranks = {}
    all_gpu_ranks = {}

    for k in parallelization_specs:
        world_sizes[k] = len(parallelization_specs[k]["gpu_ranks"])
        tensor_model_parallel_group_sizes[k] = parallelization_specs[k]["tensor_model_parallel_group_size"]
        data_parallel_group_sizes[k] = parallelization_specs[k]["data_parallel_group_size"]
        pipeline_model_parallel_group_sizes[k] = parallelization_specs[k]["pipeline_model_parallel_group_size"]
        all_num_tensor_model_parallel_groups[k] = world_sizes[k] // tensor_model_parallel_group_sizes[k]
        all_num_pipeline_components_parallel_groups[k] = world_sizes[k] // pipeline_model_parallel_group_sizes[k]
        all_num_data_parallel_groups[k] = world_sizes[k] // data_parallel_group_sizes[k]

        all_data_parallel_group_ranks[k] = []
        all_gpu_ranks[k] = parallelization_specs[k]['gpu_ranks']

    for k in parallelization_specs:
        if world_sizes[k] % (tensor_model_parallel_group_sizes[k] * pipeline_model_parallel_group_sizes[k]) != 0:
            raise RuntimeError(
                f"component world_size ({world_sizes[k]}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_group_sizes[k]}) x pipeline_model_parallel_size ({pipeline_model_parallel_group_sizes[k]})"
            )

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

    for k in parallelization_specs:
        for i in range(pipeline_model_parallel_group_sizes[k]):
            start_rank = i * all_num_pipeline_components_parallel_groups[k]
            end_rank = (i + 1) * all_num_pipeline_components_parallel_groups[k]
            for j in range(tensor_model_parallel_group_sizes[k]):
                ranks = range(all_gpu_ranks[k][start_rank + j], all_gpu_ranks[k][end_rank-1]+1, tensor_model_parallel_group_sizes[k])
                all_data_parallel_group_ranks[k].append(list(ranks))
                group = torch.distributed.new_group(ranks)
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    _DATA_PARALLEL_GROUP = group
                    _DATA_PARALLEL_GROUP_GLOO = group_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS = ranks
    first_component_name = list(parallelization_specs.keys())[0]
    last_component_name = list(parallelization_specs.keys())[-1]
    all_component_names = list(parallelization_specs.keys())


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

    # create graph connecting all gpu ranks based on pipeline_parallel groups

    list_all_num_pipeline_components_parallel_groups = list(all_num_pipeline_components_parallel_groups.values())

    # dict that stores all node to node pipeline parallel mappings for forward pass
    global _FWD_MAPPINGS
    assert len(_FWD_MAPPINGS) == 0
    fwd_mappings = {}

    global _ALL_RANKS_MICRO_BATCH_SIZES
    assert len(_ALL_RANKS_MICRO_BATCH_SIZES) == 0

    # iterate through each pipeline component parallel group
    for i, k in enumerate(all_num_pipeline_components_parallel_groups):
        
        # iterate through each rank
        for gpu_rank in all_gpu_ranks[k]:
            
            # set micro_batch_size if first component
            if k == first_component_name:
                set_ranks_micro_batch_size(gpu_rank, micro_batch_size)

            fwd_mappings[str(gpu_rank)] = []
            next_gpu_rank = gpu_rank + all_num_pipeline_components_parallel_groups[k]
            # gpu_rank mapping is within component
            # define forward mapping from current rank to next rank and
            # define the next rank's micro_batch_size
            if next_gpu_rank in all_gpu_ranks[k]:
                fwd_mappings[str(gpu_rank)].append((gpu_rank + all_num_pipeline_components_parallel_groups[k], get_ranks_micro_batch_size(gpu_rank)))
                set_ranks_micro_batch_size(next_gpu_rank, get_ranks_micro_batch_size(gpu_rank))
            # gpu_rank mapping is in the next component
            else:
                # find the index within the respective data parallel group
                data_parallel_group_index = None
                for data_parallel_group_ranks in all_data_parallel_group_ranks[k]:
                    if gpu_rank in data_parallel_group_ranks:
                        data_parallel_group_index = data_parallel_group_ranks.index(gpu_rank)

                assert data_parallel_group_index is not None
                tensor_parallel_group_index = gpu_rank % tensor_model_parallel_group_sizes[k]

                    # ---------------------------------------------------------------------------------------------------
                    # create component groups and define how data splits during fan-in/out
                    # ---------------------------------------------------------------------------------------------------
                    # walkthrough example:
                    #     first component has data parallel group size of 3, pipeline parallel group size of 1
                    #     second component has data parallel group size of 2, pipeline parallel group size of 1
                    #     visual:
                    #
                    #       component 1   component 2
                    #     ------------------------------
                    #         | 0 |          | 3 |
                    #         | 1 |          | 4 |
                    #         | 2 |
                    #     ------------------------------
                    #     step 1 (create buckets):
                    #         FORMULA: b_i = i * ratio, 
                    #             where ratio is the num pipeline_component_parallel_groups of
                    #             the next component over current component
                    #         divide range [0,2] into 3 buckets, b_1 = [0, 0.66], b_2 = [0.66, 1.33], and b_3 = [1.33, 2]
                    #     step 2 (create clean buckets):
                    #         FORMULA: [x, y] -> [floor(x), ceil(y))
                    #         the resulting buckets will be b_clean_1 = [0,1), b_clean_2 = [0, 2), b_clean_3 = [1,2)
                    #     using the buckets, 
                    #     step 3 (define component groups and portion of data to send through each group):
                    #         FORMULA: gpu_rank i connects to all elements in b_clean_i
                    #
                    #         gpu_rank 0 will connect to all elements in b_clean_1 (`[0, 1)`) of component 2, so gpu_rank 3
                    #         gpu_rank 1 will connect to all elements in b_clean_2 (`[0, 2)`) of component 2, so gpu_rank 3 and 4
                    #         gpu_rank 2 will connect to all elements in b_clean_3 (`[1, 2)`) of component 2, so gpu_rank 4
                    #
                    #         all component groupings: [0, 3], [1, 3], [1, 4], [2, 4]
                    #         now, since gpu_rank 0 (send) -> gpu_rank 3 (recv) and gpu_rank 2 (send) -> gpu_rank 4 (recv),
                    #         the the entire data slices will transfer from (send) to (recv).
                    #         for gpu_rank 1 (send), this shares its data between gpu_ranks between 3 and 4.
                    #
                    #         step 3.5 (calculate data_split_ratio)
                    #         FORMULA: min(i+1, b_j[1]) - max(i, b_j[0]) / (b_j[1]-b_j[0]) for i in range(b_clean_j)
                    #         To calculate the data_split_ratio, iterate over redefined bucket, in this case 2 iterations
                    #         1) first iteration, i=0. Take subsection of redefined bucket, [0, 1)
                    #         2) calculate overlap between [0, 1) and the original bucket, [0.66, 1.33]:
                    #             min(1, 1.33) - max(0, 0.66) = 1 - 0.66 = 0.33
                    #         3) divide overlap by size of original bucket: 0.33 / (1.33-0.66) = 0.33 / 0.66 = 1/2
                    #         This means that for component 1->3, the data_split_ratio will be 1/2. This process can be repeated
                    #         For the second iteration to get that the data_split_ratio for component 1->4 is also 1/2
                    #
                    #     final component groups with data_split_ratios
                    #     gpu_rank 0->3, ratio: 1, gpu_rank 1->3, ratio: 1/2, gpu_rank 1->4, ratio: 1/2, gpu_rank 2->4, 1
                    #     each receiving gpu_rank will recieve the same amount of data as the other receiving gpu_ranks
                    # ---------------------------------------------------------------------------------------------------

                if k != last_component_name:
                    # step 1 (define the bucket)
                    ratio = list_all_num_pipeline_components_parallel_groups[i+1] / list_all_num_pipeline_components_parallel_groups[i]
                    start = ((data_parallel_group_index)) * ratio
                    end = ((data_parallel_group_index+1)) * ratio

                    # step 2 (redefine bucket)
                    floor_start = int(math.floor((data_parallel_group_index) * ratio))
                    ceil_end = int(math.ceil(((data_parallel_group_index+1)) * ratio))

                    # steps 3 (define compoennt groups within redefined bucket)
                    new_nodes = []
                    for node in range(floor_start, ceil_end):

                        data_split_ratio = 1
                        if ceil_end-floor_start > 1:
                            divisor = end - start
                            overlap = min(node+1, end) - max(node, start)
                            data_split_ratio = overlap / divisor
                        new_node = [
                            list(all_data_parallel_group_ranks.values())[i+1][tensor_parallel_group_index][node],
                            data_split_ratio * get_ranks_micro_batch_size(gpu_rank)
                        ]
                        new_nodes.append(new_node)

                    # step 4 (define micro_batch_size for each foward mapping)
                    if len(new_nodes) > 1:
                        # fan-in/out will not always result in integer micro_batch_sizes.
                        # 1) round down each non-integer micro_batch_size and add round down amount to remainder
                        # 2) distribute remainder microbatches to the mappings with the smallest micro_batch_size
                        remainder = 0
                        for new_node in new_nodes:
                            int_batch_size = 0

                            # use math.isclose since there may be a precision issue
                            if math.isclose(new_node[1], round(new_node[1])):
                                new_node[1] = round(new_node[1])
                            else:
                                int_batch_size = int(new_node[1])
                                remainder += new_node[1] - int_batch_size
                                new_node[1] = int_batch_size
                        # distribute remaining microbatches
                        while remainder > 0:
                            smallest_micro_batch_size = math.inf
                            index_for_smallest_micro_batch_size = -1
                            for index, new_node in enumerate(new_nodes):
                                if new_node[1] < smallest_micro_batch_size:
                                    smallest_micro_batch_size = new_node[1]
                                    index_for_smallest_micro_batch_size = index

                            new_nodes[index_for_smallest_micro_batch_size][1] += 1
                            remainder -= 1

                    else:
                        new_nodes[0][1] = get_ranks_micro_batch_size(gpu_rank)

                    # add new nodes to fwd_mapping
                    for new_node in new_nodes:
                        set_ranks_micro_batch_size(new_node[0], new_node[1])
                        fwd_mappings[str(gpu_rank)].append(new_node)
                else:
                    if len(fwd_mappings[str(gpu_rank)]) == 0:
                        set_ranks_micro_batch_size(gpu_rank, get_ranks_micro_batch_size(gpu_rank-all_num_tensor_model_parallel_groups[k]))

    _FWD_MAPPINGS = fwd_mappings
    assert len(_ALL_RANKS_MICRO_BATCH_SIZES) == sum(world_sizes.values()), \
        "Something went wrong with defining _ALL_RANKS_MICRO_BATCH_SIZES, please file a bug."

    # create same mapping but for backward step
    global _BKWD_MAPPINGS
    assert len(_BKWD_MAPPINGS) == 0
    bkwd_mappings = {}

    reversed_all_num_pipeline_components_parallel_groups = dict(reversed(list(all_num_pipeline_components_parallel_groups.items())))
    reversed_list_all_num_pipeline_components_parallel_groups = list(reversed(list_all_num_pipeline_components_parallel_groups))
    reversed_tensor_model_parallel_group_sizes = dict(reversed(list(tensor_model_parallel_group_sizes.items())))
    
    reversed_all_gpu_ranks = dict(reversed(list(all_gpu_ranks.items())))
    reversed_all_data_parallel_group_ranks = dict(reversed(all_data_parallel_group_ranks.items()))
    for k in reversed_all_gpu_ranks:
        reversed_all_gpu_ranks[k] = list(reversed(list(reversed_all_gpu_ranks[k])))
        reversed_all_data_parallel_group_ranks[k] = list(reversed(reversed_all_data_parallel_group_ranks[k]))
        for i, _ in enumerate(reversed_all_data_parallel_group_ranks[k]):
            reversed_all_data_parallel_group_ranks[k][i] = list(reversed(reversed_all_data_parallel_group_ranks[k][i]))

    # iterate through each pipeline component parallel group
    for i, k in enumerate(reversed_all_num_pipeline_components_parallel_groups):
        
        # iterate through each rank
        for gpu_rank in reversed_all_gpu_ranks[k]:
            bkwd_mappings[str(gpu_rank)] = []
            # gpu_rank mapping is within component
            if (gpu_rank - all_num_pipeline_components_parallel_groups[k]) in all_gpu_ranks[k]:
                bkwd_mappings[str(gpu_rank)].append((gpu_rank - all_num_pipeline_components_parallel_groups[k], get_ranks_micro_batch_size(gpu_rank)))
            # gpu_rank mappin is in the next component
            else:
                # find the index within the respective data parallel group
                data_parallel_group_index = None
                for data_parallel_group_ranks in reversed_all_data_parallel_group_ranks[k]:
                    if gpu_rank in data_parallel_group_ranks:
                        data_parallel_group_index = data_parallel_group_ranks.index(gpu_rank)
                
                assert data_parallel_group_index is not None
                tensor_parallel_group_index = (reversed_tensor_model_parallel_group_sizes[k]-1) - (gpu_rank) % reversed_tensor_model_parallel_group_sizes[k]

                if k != first_component_name:
                    # step 1 (define the bucket)
                    ratio = reversed_list_all_num_pipeline_components_parallel_groups[i+1] / reversed_list_all_num_pipeline_components_parallel_groups[i]
                    start = ((data_parallel_group_index)) * ratio
                    end = ((data_parallel_group_index+1)) * ratio

                    # step 2 (redefine bucket)
                    floor_start = int(math.floor((data_parallel_group_index) * ratio))
                    ceil_end = int(math.ceil(((data_parallel_group_index+1)) * ratio))
                    
                    # steps 3 (define compoennt groups within redefined bucket)
                    new_nodes = []
                    for node in range(floor_start, ceil_end):
                        destination = list(reversed_all_data_parallel_group_ranks.values())[i+1][tensor_parallel_group_index][node]
                        partial_micro_batch_size = None
                        for path in fwd_mappings[str(destination)]:
                            if gpu_rank == path[0]:
                                partial_micro_batch_size = path[1]
                        assert partial_micro_batch_size != None
                        new_node = [
                            destination,
                            partial_micro_batch_size
                        ]
                        new_nodes.append(new_node.copy())
                        
                    # add new nodes to fwd_mapping
                    for new_node in new_nodes:
                        bkwd_mappings[str(gpu_rank)].append(new_node)

    _BKWD_MAPPINGS = bkwd_mappings
    # dfs to create pipeline model and component paralel ranks
    visited_paths = {}

    # dictionaries to track initialized embedding and position embedding ranks
    existing_embedding_ranks = {}
    existing_position_embedding_ranks = {}

    list_connector_ranks = {}

    def pipeline_parallelism_setup(curr_rank):
        
        # check if dfs have reached end of pipeline_model_parallel_group
        if len(fwd_mappings[str(curr_rank)]) == 0:
            # initialize pipeline_model_parallel_group_here
            group = torch.distributed.new_group(pipeline_model_parallel_ranks.copy())
            if rank in pipeline_model_parallel_ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                _PIPELINE_GLOBAL_RANKS.append(pipeline_model_parallel_ranks.copy())

                # define first and last ranks in full model pipeline parallel group
                _FIRST_PIPELINE_MODEL_PARALLEL_RANK = pipeline_model_parallel_ranks[0]
                _LAST_PIPELINE_MODEL_PARALLEL_RANK = pipeline_model_parallel_ranks[-1]

                if rank != pipeline_model_parallel_ranks[0]:
                    _PREV_PIPELINE_MODEL_PARALLEL_RANK.append(pipeline_model_parallel_ranks[pipeline_model_parallel_ranks.index(rank) - 1])
                if rank != pipeline_model_parallel_ranks[-1]:
                    _NEXT_PIPELINE_MODEL_PARALLEL_RANK.append(pipeline_model_parallel_ranks[pipeline_model_parallel_ranks.index(rank) + 1])
                    # Setup embedding group (to exchange gradients between

            # first and last stages).
            if len(pipeline_model_parallel_ranks) > 1:
                embedding_ranks = [pipeline_model_parallel_ranks[0], pipeline_model_parallel_ranks[-1]]
                position_embedding_ranks = [pipeline_model_parallel_ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if pipeline_model_parallel_ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [pipeline_model_parallel_ranks[0],
                                        pipeline_model_parallel_ranks[pipeline_model_parallel_split_rank],
                                        pipeline_model_parallel_ranks[-1]]
                    if pipeline_model_parallel_ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [pipeline_model_parallel_ranks[0],
                                        pipeline_model_parallel_ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = pipeline_model_parallel_ranks
                position_embedding_ranks = pipeline_model_parallel_ranks

            if str(embedding_ranks) not in existing_embedding_ranks:
                existing_embedding_ranks[str(embedding_ranks)] = ''
                group = torch.distributed.new_group(embedding_ranks)
                if rank in embedding_ranks:
                    _EMBEDDING_GROUP.append(group)
                if rank in pipeline_model_parallel_ranks:
                    _EMBEDDING_GLOBAL_RANKS.append(embedding_ranks)

            if str(position_embedding_ranks) not in existing_position_embedding_ranks:
                existing_position_embedding_ranks[str(position_embedding_ranks)] = ''
                group = torch.distributed.new_group(position_embedding_ranks)
                if rank in position_embedding_ranks:
                    _POSITION_EMBEDDING_GROUP.append(group)
                if rank in pipeline_model_parallel_ranks:
                    _POSITION_EMBEDDING_GLOBAL_RANKS.append(position_embedding_ranks)

        # add current path to visited_paths
        visited_paths[str(list(pipeline_model_parallel_ranks))] = ''
        # visit all possible fwd_mappings in a depth-first search manner if not already visited
        for next_rank, _ in fwd_mappings[str(curr_rank)]:
            if next_rank not in model_parallel_group:
                model_parallel_group.append(next_rank)
            pipeline_model_parallel_ranks.append(next_rank)
            if str(list(pipeline_model_parallel_ranks)) not in visited_paths:
                pipeline_parallelism_setup(next_rank)
            pipeline_model_parallel_ranks.pop()


    # Build the pipeline model-parallel, component-parallel groups,
    # model-parallel groups, and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _FIRST_PIPELINE_MODEL_PARALLEL_RANK
    global _LAST_PIPELINE_MODEL_PARALLEL_RANK
    global _PREV_PIPELINE_MODEL_PARALLEL_RANK
    global _NEXT_PIPELINE_MODEL_PARALLEL_RANK

    assert not len(_PIPELINE_MODEL_PARALLEL_GROUP), \
        'pipeline model parallel group is already initialized'

    global _MODEL_PARALLEL_GROUP
    assert len(_MODEL_PARALLEL_GROUP) == 0, 'model parallel group is already initialized'
    
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert len(_EMBEDDING_GROUP) == 0, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert len(_POSITION_EMBEDDING_GROUP) == 0, \
        'position embedding group is already initialized'
    # iterate through all starting ranks, define pipeline_model_parallel_groups and model_parallel_groups
    start_rank = 0
    model_parallel_group = []
    while start_rank < (data_parallel_group_sizes[first_component_name] * tensor_model_parallel_group_sizes[first_component_name]):
        # need to fix for v2, this asusmes that tensor_model_parallel_group_sizes is uniform across components
        if start_rank > 0 and start_rank % tensor_model_parallel_group_sizes[first_component_name] == 0:
            model_parallel_group.sort()
            group = torch.distributed.new_group(model_parallel_group)
            if rank in model_parallel_group:
                _MODEL_PARALLEL_GROUP.append(group)
            model_parallel_group = [start_rank]
        else:
            model_parallel_group.append(start_rank)
        pipeline_model_parallel_ranks = [start_rank]
        pipeline_parallelism_setup(start_rank)
        start_rank += 1

    model_parallel_group.sort()
    group = torch.distributed.new_group(model_parallel_group)
    if rank in model_parallel_group:
        _MODEL_PARALLEL_GROUP.append(group)


    # define pipeline component parallel ranks (no need to initialize)
    global _PIPELINE_COMPONENT_GLOBAL_RANKS
    assert _PIPELINE_COMPONENT_GLOBAL_RANKS is None
    
    # components will always be the same, regardless of the number of different pipeline model parallel groups for a specific rank
    _PIPELINE_COMPONENT_GLOBAL_RANKS = []
    for component_rank in get_pipeline_model_parallel_groups_ranks()[0]:
        for k in parallelization_specs:
            if rank in all_gpu_ranks[k] and component_rank in all_gpu_ranks[k]:
                _PIPELINE_COMPONENT_GLOBAL_RANKS.append(component_rank)


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


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_using_layer_unit_test_strategy():
    assert _USING_LAYER_UNIT_TEST_STRATEGY is not None, \
        'parallel state not intialized'
    return _USING_LAYER_UNIT_TEST_STRATEGY


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert len(_MODEL_PARALLEL_GROUP) != 0, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP[0]


def get_model_parallel_groups():
    """Get the model parallel group the caller rank belongs to."""
    assert len(_MODEL_PARALLEL_GROUP) != 0, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

# TODO (gersonkroiz): change this to plural
def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert len(_PIPELINE_MODEL_PARALLEL_GROUP) != 0 , \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP[0]

def get_pipeline_model_parallel_groups():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert len(_PIPELINE_MODEL_PARALLEL_GROUP) != 0 , \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_component_parallel_groups_ranks(direction: Optional[str] = ''):
    """Get the pipeline component parallel group the caller rank belongs to."""
    """If direction is specific, get the next/prev component parallel connector group"""
    """Will return a list of pipeline component parallel group ranks"""
    if direction != '':
        assert direction in ['next', 'prev']
        if direction == 'next':
            assert not is_pipeline_last_stage()
            return get_pipeline_model_parallel_next_ranks()
        if direction == 'prev':
            assert not is_pipeline_first_stage()
            return get_pipeline_model_parallel_prev_ranks()
    assert len(_PIPELINE_COMPONENT_GLOBAL_RANKS) != 0, \
        'pipeline_model parallel group is not defined'
    return _PIPELINE_COMPONENT_GLOBAL_RANKS


def get_pipeline_model_parallel_groups_ranks():
    assert len(_PIPELINE_GLOBAL_RANKS) > 0
    return _PIPELINE_GLOBAL_RANKS


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
    assert len(_EMBEDDING_GROUP) != 0, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP[0]


def get_embedding_groups():
    """Get the embedding group the caller rank belongs to."""
    assert len(_EMBEDDING_GROUP) != 0, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert len(_POSITION_EMBEDDING_GROUP) != 0, \
        'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP[0]


def get_position_embedding_groups():
    """Get the position embedding group the caller rank belongs to."""
    assert len(_POSITION_EMBEDDING_GROUP) != 0, \
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
    assert len(_PIPELINE_COMPONENT_GLOBAL_RANKS) != 0
    return len(_PIPELINE_COMPONENT_GLOBAL_RANKS)


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
    # all pipieline model parallel groups are the same size and
    # have rank at the same index, so we can just use the first group
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_component_parallel_rank():
    """Return my rank for the pipeline component parallel group."""
    global _MPU_PIPELINE_COMPONENT_PARALLEL_RANK
    if _MPU_PIPELINE_COMPONENT_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_COMPONENT_PARALLEL_RANK
    # all pipieline component parallel groups are the same size and
    # have rank at the same index, so we can just use the first group
    rank = rank = torch.distributed.get_rank()
    return _PIPELINE_COMPONENT_GLOBAL_RANKS.index(rank)


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
    assert len(_EMBEDDING_GLOBAL_RANKS) == 1
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS[0]
    if rank in _EMBEDDING_GLOBAL_RANKS[0]:
        if rank == _EMBEDDING_GLOBAL_RANKS[0][0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[0][-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert len(_POSITION_EMBEDDING_GLOBAL_RANKS) == 1
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS[0]


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

# calls to this need adjustment for LUTS
def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    if get_using_layer_unit_test_strategy():
        assert len(_FIRST_PIPELINE_MODEL_PARALLEL_RANK) != 0, \
            "First pipeline parallel model rank is not initialized"
        return _FIRST_PIPELINE_MODEL_PARALLEL_RANK
    else:
        assert len(_PIPELINE_GLOBAL_RANKS) == 1, \
            "Pipeline parallel group is not initialized"
        return _PIPELINE_GLOBAL_RANKS[0][0]

# calls to this need adjustment for LUTS
def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    if get_using_layer_unit_test_strategy():
        assert len(_LAST_PIPELINE_MODEL_PARALLEL_RANK) != 0, \
            "Last pipeline parallel model rank is not initialized"
        return _LAST_PIPELINE_MODEL_PARALLEL_RANK
    else:
        assert len(_PIPELINE_GLOBAL_RANKS) == 1, \
            "Pipeline parallel group is not initialized"
        last_rank_local = get_pipeline_model_parallel_world_size() - 1
        return _PIPELINE_GLOBAL_RANKS[0][last_rank_local]

# calls to this need adjustment for LUTS
def get_pipeline_model_parallel_next_ranks():
    """Return the global rank that follows the caller in the pipeline"""
    if get_using_layer_unit_test_strategy():
        assert len(_NEXT_PIPELINE_MODEL_PARALLEL_RANK) != 0, \
            "Previous pipeline parallel group is not initialized"
        return _NEXT_PIPELINE_MODEL_PARALLEL_RANK
    else:
        assert len(_PIPELINE_GLOBAL_RANKS) == 1, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        return _PIPELINE_GLOBAL_RANKS[0][(rank_in_pipeline + 1) % world_size]

# calls to this need adjustment for LUTS
def get_pipeline_model_parallel_prev_ranks():
    """Return the global rank that preceeds the caller in the pipeline"""
    if get_using_layer_unit_test_strategy():
        assert len(_PREV_PIPELINE_MODEL_PARALLEL_RANK) != 0, \
            "Previous pipeline parallel group is not initialized"
        return _PREV_PIPELINE_MODEL_PARALLEL_RANK
    else:
        assert len(_PIPELINE_GLOBAL_RANKS) == 1, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        return _PIPELINE_GLOBAL_RANKS[0][(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_full_mapping(direction: str):
    """Return either full forward or backward mappings based on function input"""
    if direction == "fwd":
        assert len(_FWD_MAPPINGS) != 0
        return _FWD_MAPPINGS
    elif direction == "bkwd":
        assert len(_BKWD_MAPPINGS) != 0
        return _BKWD_MAPPINGS
    raise ValueError("direction must be either `bkwd` or `fwd`")
    

def get_rank_mapping(direction: str):
    """Return ranks that current rank maps to, dependent on function input"""
    rank = torch.distributed.get_rank()
    if direction == "fwd":
        assert len(_FWD_MAPPINGS) != 0
        return _FWD_MAPPINGS[str(rank)]
    elif direction == "bkwd":
        assert len(_BKWD_MAPPINGS) != 0
        return _BKWD_MAPPINGS[str(rank)]
    raise ValueError("direction must be either `bkwd` or `fwd`")


def set_ranks_micro_batch_size(rank: int, micro_batch_size: int):
    if rank not in _ALL_RANKS_MICRO_BATCH_SIZES:
        _ALL_RANKS_MICRO_BATCH_SIZES[rank] = micro_batch_size
    else:
        _ALL_RANKS_MICRO_BATCH_SIZES[rank] += micro_batch_size


def get_ranks_micro_batch_size(rank: int):
    return _ALL_RANKS_MICRO_BATCH_SIZES[rank]


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
    _MODEL_PARALLEL_GROUP = []
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = []
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = []
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = []
    global _EMBEDDING_GLOBAL_RANKS
    _EMBEDDING_GLOBAL_RANKS = []
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    _POSITION_EMBEDDING_GLOBAL_RANKS = []
    global _AMAX_REDUCTION_GROUP
    _AMAX_REDUCTION_GROUP = None
    global _PREV_PIPELINE_MODEL_PARALLEL_RANK
    _PREV_PIPELINE_MODEL_PARALLEL_RANK = []
    global _NEXT_PIPELINE_MODEL_PARALLEL_RANK
    _NEXT_PIPELINE_MODEL_PARALLEL_RANK = []
    global _FIRST_PIPELINE_MODEL_PARALLEL_RANK
    _FIRST_PIPELINE_MODEL_PARALLEL_RANK = []
    global _LAST_PIPELINE_MODEL_PARALLEL_RANK
    _LAST_PIPELINE_MODEL_PARALLEL_RANK = []
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
    global _USING_LAYER_UNIT_TEST_STRATEGY
    _USING_LAYER_UNIT_TEST_STRATEGY = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = []
    global _PIPELINE_COMPONENT_GLOBAL_RANKS
    _PIPELINE_COMPONENT_GLOBAL_RANKS = None
    global _FWD_MAPPINGS
    _FWD_MAPPINGS = {}
    global _BKWD_MAPPINGS
    _BKWD_MAPPINGS = {}
    global _ALL_RANKS_MICRO_BATCH_SIZES
    _ALL_RANKS_MICRO_BATCH_SIZES = {}