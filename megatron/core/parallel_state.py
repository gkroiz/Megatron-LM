# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
import warnings
from datetime import timedelta
from typing import List, Optional

import torch

# from .utils import GlobalMemoryBuffer

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUPS = []
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUPS = []
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None


_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = []

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# MOE logging
_MOE_AUX_LOSSES_LOGGING_TRACKER = {}

###### Objects for supporting non-uniform parallelism ######

# Previous component pipeline group that the current rank belongs to.
_PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS = []
# Next component pipeline group that the current rank belongs to.
_NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS = []

# Previous pipeline model rank based on current rank.
_PREV_PIPELINE_MODEL_PARALLEL_RANKS = []
# Next pipeline model rank based on current rank.
_NEXT_PIPELINE_MODEL_PARALLEL_RANKS = []
# First pipeline model rank based on current rank.
_FIRST_PIPELINE_MODEL_PARALLEL_RANKS = []
# Last pipeline model rank based on current rank.
_LAST_PIPELINE_MODEL_PARALLEL_RANKS = []

# number of layers in the component that the current rank belongs to
_NUM_COMPONENT_LAYERS = None

# whether layer unit test strategy is used (boolean)
_USING_LAYER_UNIT_TEST_STRATEGY = None

# ratio for fan-in fan-out (assuming stimulus and response components have same number of data parallel groups)
_FIFO_RATIO = None

# booleans for which component current rank is in
_IS_RANK_IN_FIRST_COMPONENT = False
_IS_RANK_IN_LAST_COMPONENT = False

############################################################


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool],
) -> List[List[int]]:
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator(object):
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        if 'ep' in order:
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        print(f'{order=}')
        print(f'{parallel_size=}')
        print(f'{mask=}')
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)
        return ranks


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
) -> None:
    """Initialize model data parallel groups.

    Args:
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

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.
        
        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

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
    global _USING_LAYER_UNIT_TEST_STRATEGY
    _USING_LAYER_UNIT_TEST_STRATEGY = False

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'

    for ranks in rank_generator.get_ranks('dp'):
        group = rankstorch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        group = ranks
        group_gloo = ranks
        if rank in ranks:
            _DATA_PARALLEL_GROUP = ranks
            _DATA_PARALLEL_GROUP_GLOO = ranks
            _DATA_PARALLEL_GLOBAL_RANKS = ranks
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        group_with_cp_gloo = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, backend="gloo"
        )
        if rank in ranks_with_cp:
            _DATA_PARALLEL_GROUP_WITH_CP = ranks_with_cp
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO = ranks_with_cp
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for ranks in rank_generator.get_ranks('cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUPS
    assert len(_MODEL_PARALLEL_GROUPS) == 0, 'model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUPS.append(ranks)

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = ranks
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUPS
    global _PIPELINE_GLOBAL_RANKS
    assert (
        len(_PIPELINE_MODEL_PARALLEL_GROUPS) == 0
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    for ranks in rank_generator.get_ranks('pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUPS.append(ranks)
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = embedding_ranks
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = position_embedding_ranks
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    for ranks in rank_generator.get_ranks('tp-dp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = ranks
    for ranks in rank_generator.get_ranks('tp-dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP = ranks

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO

    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_EXPERT_PARALLEL_GROUP = ranks

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = ranks

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            _DATA_MODULO_EXPERT_PARALLEL_GROUP = ranks
            _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = ranks

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def initialize_model_components_parallel(
    parallelization_specs: dict,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
):
    """Initialize model data parallel groups for each component of the model.

    Args:
        parallelization_specs: (dict, required)
            contains model component-specific parallelization

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

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.
        
        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

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
    global _USING_LAYER_UNIT_TEST_STRATEGY
    _USING_LAYER_UNIT_TEST_STRATEGY = True
    
    assert context_parallel_size == 1, \
        "`initialize_model_components_parallel()` does not support `context_parallel_size > 1`"
    assert expert_model_parallel_size == 1, \
        "`initialize_model_components_parallel()` does not support `expert_model_parallel_size > 1`"

    # Get world size and rank. Ensure some consistencies.
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
                f"component world_size ({world_sizes[k]}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_group_sizes[k]}) x pipeline_model_parallel_size ({pipeline_model_parallel_group_sizes[k]})"
            )

    for k in parallelization_specs:
        if (
            world_sizes[k]
            % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
            != 0
        ):
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
                f"x context_parallel_size ({context_parallel_size})"
            )

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
    )
    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'

    for k in parallelization_specs:
        for i in range(pipeline_model_parallel_group_sizes[k]):
            start_rank = i * all_num_pipeline_model_parallel_groups[k]
            end_rank = (i + 1) * all_num_pipeline_model_parallel_groups[k]
            for j in range(tensor_model_parallel_group_sizes[k]):
                ranks = range(all_gpu_ranks[k][start_rank + j], all_gpu_ranks[k][end_rank-1]+1, tensor_model_parallel_group_sizes[k])
                all_data_parallel_group_ranks[k].append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
                if rank in ranks:
                    _DATA_PARALLEL_GROUP = group
                    _DATA_PARALLEL_GROUP_GLOO = group_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS = ranks
                    
    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # define global variables for which component rank is in
    global _IS_RANK_IN_FIRST_COMPONENT
    global _IS_RANK_IN_LAST_COMPONENT
    if rank in all_gpu_ranks[first_component_name]:
        _IS_RANK_IN_FIRST_COMPONENT = True
    if rank in all_gpu_ranks[last_component_name]:
        _IS_RANK_IN_LAST_COMPONENT = True

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUPS
    assert len(_MODEL_PARALLEL_GROUPS) == 0, 'model parallel group is already initialized'

    # assert that number of data parallel groups 
    assert data_parallel_group_sizes[first_component_name] == data_parallel_group_sizes[last_component_name]

    #        |          |              |          |
    # assert | Stimulus | > | Test | < | Response | (fan-in to fan-out structure)
    #        |          |              |          |
    assert data_parallel_group_sizes[first_component_name] >= data_parallel_group_sizes[middle_component_name]
    assert data_parallel_group_sizes[last_component_name] >= data_parallel_group_sizes[middle_component_name]

    # assert evenly divisble fan-in/out
    assert data_parallel_group_sizes[first_component_name] % data_parallel_group_sizes[middle_component_name] == 0
    ratio = data_parallel_group_sizes[first_component_name] // data_parallel_group_sizes[middle_component_name]

    global _FIFO_RATIO
    _FIFO_RATIO = ratio

    # for each different data parallel group in the first component,
    # define a model-parallel group
    for i in range(data_parallel_group_sizes[first_component_name]):
        ranks = []
        for k in parallelization_specs:
            for data_parallel_group_ranks in all_data_parallel_group_ranks[k]:
                # adjust index based on pre-defined ratio
                if k == middle_component_name:
                    ranks.append(data_parallel_group_ranks[i // ratio])
                else:
                    ranks.append(data_parallel_group_ranks[i])
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUPS.append(group)

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for k in parallelization_specs:
        for i in range(all_num_tensor_model_parallel_groups[k]):
            ranks = range(all_gpu_ranks[k][i * tensor_model_parallel_group_sizes[k]],
                          all_gpu_ranks[k][((i + 1) * tensor_model_parallel_group_sizes[k])-1]+1)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel and component pipeline connector groups
    global _PIPELINE_COMPONENT_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUPS
    global _PIPELINE_GLOBAL_RANKS
    global _PIPELINE_COMPONENT_GLOBAL_RANKS
    global _PREV_PIPELINE_MODEL_PARALLEL_RANKS
    global _NEXT_PIPELINE_MODEL_PARALLEL_RANKS
    global _FIRST_PIPELINE_MODEL_PARALLEL_RANKS
    global _LAST_PIPELINE_MODEL_PARALLEL_RANKS
    global _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS
    global _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS
    global _PREV_COMPONENT_PIPELINE_GLOBAL_RANKS
    global _NEXT_COMPONENT_PIPELINE_GLOBAL_RANKS

    # arrays to define full pipeline model parallel groups
    full_pipeline_model_parallel_groups = [[] for _ in range(all_num_pipeline_model_parallel_groups[first_component_name])]
    full_pipeline_model_parallel_groups_ranks_tracker = [_ for _ in range(all_num_pipeline_model_parallel_groups[first_component_name])]

    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'

    # since tensor model parallelism is uniform
    tensor_model_parallel_group_size = tensor_model_parallel_group_sizes[first_component_name]

    # data_parallel_group_sizes[first_component_name] * tensor_model_parallel_group_size = number of model pipeline parallel groups
    # TODO (gkroiz): change for non-uniform tensor parallelism
    for i in range(data_parallel_group_sizes[first_component_name]):
        for j in range(tensor_model_parallel_group_size):
            pipeline_model_parallel_ranks = []

            # iterate through each component
            for k in parallelization_specs:
                pipeline_component_parallel_ranks = []

                # define next rank in pipeline component group
                for data_parallel_groups_index, data_parallel_group_ranks in enumerate(all_data_parallel_group_ranks[k]):
                    if data_parallel_groups_index % tensor_model_parallel_group_size == j:
                        if k == middle_component_name:
                            pipeline_component_parallel_ranks.append(data_parallel_group_ranks[i // ratio])
                        else:
                            pipeline_component_parallel_ranks.append(data_parallel_group_ranks[i])

                # define component group
                pipeline_component_parallel_group = torch.distributed.new_group(
                    pipeline_component_parallel_ranks,
                    timeout=timeout,
                    pg_options=get_nccl_options('pp', nccl_comm_cfgs)
                )
                if rank in pipeline_component_parallel_ranks:
                    _PIPELINE_COMPONENT_PARALLEL_GROUP = pipeline_component_parallel_group
                    _PIPELINE_COMPONENT_GLOBAL_RANKS = pipeline_component_parallel_ranks

                # define connector ranks
                if len(pipeline_model_parallel_ranks) != 0:
                    connector_ranks = [pipeline_model_parallel_ranks[-1], pipeline_component_parallel_ranks[0]]
                    connector_group = torch.distributed.new_group(
                        connector_ranks,
                        timeout=timeout,
                        pg_options=get_nccl_options('pp', nccl_comm_cfgs)
                    )

                    if rank in connector_ranks:
                        if rank == connector_ranks[0]:
                            _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS.append(connector_group)
                            _NEXT_COMPONENT_PIPELINE_GLOBAL_RANKS.append(connector_ranks)

                    if rank == connector_ranks[1]:
                        _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS.append(connector_group)
                        _PREV_COMPONENT_PIPELINE_GLOBAL_RANKS.append(connector_ranks)

                pipeline_model_parallel_ranks += pipeline_component_parallel_ranks

            # define pipeline model parallel group + other related global variables
            pipeline_model_parallel_group = torch.distributed.new_group(
                pipeline_model_parallel_ranks,
                timeout=timeout,
                pg_options=get_nccl_options('pp', nccl_comm_cfgs)
            )
            if rank in pipeline_model_parallel_ranks:
                _PIPELINE_MODEL_PARALLEL_GROUPS.append(pipeline_model_parallel_group)
                _PIPELINE_GLOBAL_RANKS.append(pipeline_model_parallel_ranks)

                # define previous and next ranks in pipeline model parallel group
                ranks_index = pipeline_model_parallel_ranks.index(rank)
                _PREV_PIPELINE_MODEL_PARALLEL_RANKS.append(pipeline_model_parallel_ranks[ranks_index-1] if ranks_index-1 >= 0 else -1)
                _NEXT_PIPELINE_MODEL_PARALLEL_RANKS.append(pipeline_model_parallel_ranks[ranks_index+1] if ranks_index+1 < len(pipeline_model_parallel_ranks) else -1)

                # define first and last ranks in pipeline modell parallel group
                _FIRST_PIPELINE_MODEL_PARALLEL_RANKS.append(pipeline_model_parallel_ranks[0])
                _LAST_PIPELINE_MODEL_PARALLEL_RANKS.append(pipeline_model_parallel_ranks[-1])

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

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    for k in parallelization_specs:
        amax_group_size: int = tensor_model_parallel_group_sizes[k] * data_parallel_group_sizes[k]
        num_amax_groups: int = world_sizes[k] // amax_group_size
        for i in range(num_amax_groups):
            start_rank = i * amax_group_size
            end_rank = (i + 1) * amax_group_size
            ranks = range(all_gpu_ranks[k][start_rank], all_gpu_ranks[k][end_rank-1]+1)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn(
        "is_unitialized is deprecated, use is_initialized instead", DeprecationWarning,
    )
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_using_layer_unit_test_strategy():
    assert _USING_LAYER_UNIT_TEST_STRATEGY is not None, \
        'parallel state not intialized'
    return _USING_LAYER_UNIT_TEST_STRATEGY


def get_fifo_ratio():
    assert _FIFO_RATIO is not None, \
        'first-in first-out ratio is not defined'
    return _FIFO_RATIO


def get_model_parallel_group(index=0):
    """Get the model parallel group the caller rank belongs to."""
    assert len(_MODEL_PARALLEL_GROUPS) != 0, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUPS[index]


def get_model_parallel_groups():
    """Get all model parallel groups the caller rank belongs to."""
    assert len(_MODEL_PARALLEL_GROUPS) != 0, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUPS


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group(index=0):
    """Get the pipeline model parallel group (based on index input)"""
    """the caller rank belongs to."""
    assert len(_PIPELINE_MODEL_PARALLEL_GROUPS) != 0, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUPS[index]


def get_pipeline_model_parallel_groups():
    """Get all pipeline model parallel groups the caller rank belongs to."""
    assert len(_PIPELINE_MODEL_PARALLEL_GROUPS) != 0, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUPS


def get_pipeline_component_parallel_group(direction: Optional[str] = '', index: int = 0):
    """Get the pipeline component parallel group (based on index input) the caller rank belongs to."""
    """If direction is specific, get the next/prev component parallel connector group"""
    assert get_using_layer_unit_test_strategy() is True, \
        'pipeline component parallel groups is specific to LUTS'
    if direction != '':
        assert direction in ['next', 'prev']
        if direction == 'next' and is_pipeline_component_last_stage():
            assert len(_NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS[index]
        if direction == 'prev' and is_pipeline_component_first_stage():
            assert len(_PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS[index]
    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_COMPONENT_PARALLEL_GROUP


def get_pipeline_component_parallel_groups(direction: Optional[str] = ''):
    """Get all pipeline component parallel group the caller rank belongs to."""
    """If direction is specific, get all next/prev component parallel connector group"""
    assert get_using_layer_unit_test_strategy() is True, \
        'pipeline component parallel groups is specific to LUTS'
    if direction != '':
        assert direction in ['next', 'prev']
        if direction == 'next' and is_pipeline_component_last_stage():
            assert len(_NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS
        if direction == 'prev' and is_pipeline_component_first_stage():
            assert len(_PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS
    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_COMPONENT_PARALLEL_GROUP


def get_pipeline_component_parallel_group_ranks(direction: Optional[str] = '', index: int = 0):
    """Get the pipeline component parallel ranks (based on index input) the caller rank belongs to."""
    """If direction is specific, get the next/prev component parallel connector ranks"""
    assert get_using_layer_unit_test_strategy() is True, \
        'pipeline component parallel groups is specific to LUTS'
    if direction != '':
        assert direction in ['next', 'prev']
        if direction == 'next' and is_pipeline_component_last_stage():
            assert len(_NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _NEXT_COMPONENT_PIPELINE_GLOBAL_RANKS[index]
        if direction == 'prev' and is_pipeline_component_first_stage():
            assert len(_PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _PREV_COMPONENT_PIPELINE_GLOBAL_RANKS[index]
    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_COMPONENT_GLOBAL_RANKS


def get_pipeline_component_parallel_groups_ranks(direction: Optional[str] = ''):
    """Get all pipeline component parallel ranks the caller rank belongs to."""
    """If direction is specific, get all next/prev component parallel connector ranks"""
    assert get_using_layer_unit_test_strategy() is True, \
        'pipeline component parallel groups is specific to LUTS'
    if direction != '':
        assert direction in ['next', 'prev']
        if direction == 'next' and is_pipeline_component_last_stage():
            assert len(_NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _NEXT_COMPONENT_PIPELINE_GLOBAL_RANKS
        if direction == 'prev' and is_pipeline_component_first_stage():
            assert len(_PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS) != 0
            return _PREV_COMPONENT_PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_COMPONENT_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_COMPONENT_GLOBAL_RANKS


def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_expert_model_parallel_group():
    assert (
        _EXPERT_MODEL_PARALLEL_GROUP is not None
    ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group_gloo():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is not None
    ), 'data modulo expert parallel group-gloo is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO


def set_expert_model_parallel_world_size(world_size):
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


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


def get_num_component_layers():
    """Return the number of layers in component."""
    global _NUM_COMPONENT_LAYERS
    return _NUM_COMPONENT_LAYERS


def set_num_component_layers(num_layers):
    """Set the number of layers in compoment."""
    global _NUM_COMPONENT_LAYERS
    _NUM_COMPONENT_LAYERS = num_layers


def set_expert_model_parallel_rank(rank):
    """Set expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


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
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_component_first_stage():
    """Return True if in the first pipeline component-parallel stage, False otherwise."""
    # TODO: implement virtual conditional
    return get_pipeline_component_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_pipeline_component_last_stage():
    """Return True if in the last pipeline component-parallel stage, False otherwise."""
    # TODO: implement virtual conditional
    return get_pipeline_component_parallel_rank() == (
        get_pipeline_component_parallel_world_size() - 1)


def is_rank_in_first_component():
    return _IS_RANK_IN_FIRST_COMPONENT


def is_rank_in_last_component():
    return _IS_RANK_IN_LAST_COMPONENT


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
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


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


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert (
        _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None
    ), "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank(index=0):
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group (based on input index)"""
    if get_using_layer_unit_test_strategy():
        assert len(_FIRST_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "First pipeline parallel model rank is not initialized"
        return _FIRST_PIPELINE_MODEL_PARALLEL_RANKS[index]
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_ranks():
    """Return all global rank of the first process in the pipeline for the
    current tensor parallel group"""
    if get_using_layer_unit_test_strategy():
        assert len(_FIRST_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "First pipeline parallel model rank is not initialized"
        return _FIRST_PIPELINE_MODEL_PARALLEL_RANKS
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        return [_PIPELINE_GLOBAL_RANKS[0]]


def get_pipeline_model_parallel_last_rank(index=0):
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group (based on input index)"""
    if get_using_layer_unit_test_strategy():
        assert len(_LAST_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "Last pipeline parallel model rank is not initialized"
        return _LAST_PIPELINE_MODEL_PARALLEL_RANKS[index]
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        last_rank_local = get_pipeline_model_parallel_world_size() - 1
        return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_last_ranks():
    """Return all global rank of the last process in the pipeline for the
    current tensor parallel group based on index"""
    if get_using_layer_unit_test_strategy():
        assert len(_LAST_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "Last pipeline parallel model rank is not initialized"
        return _LAST_PIPELINE_MODEL_PARALLEL_RANKS
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        last_rank_local = get_pipeline_model_parallel_world_size() - 1
        return [_PIPELINE_GLOBAL_RANKS[last_rank_local]]


def get_pipeline_model_parallel_next_rank(index=0):
    """Return the global rank that follows the caller in the pipeline"""
    if get_using_layer_unit_test_strategy():
        assert len(_NEXT_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "Next pipeline parallel model rank is not initialized"
        assert -1 not in _NEXT_PIPELINE_MODEL_PARALLEL_RANKS, \
            "Current rank does not have a a next pipeline model parallel rank"
        return _NEXT_PIPELINE_MODEL_PARALLEL_RANKS[index]
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_next_ranks():
    """Return all global rank that follows the caller in the pipeline"""
    if get_using_layer_unit_test_strategy():
        assert len(_NEXT_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "Next pipeline parallel model rank is not initialized"
        assert -1 not in _NEXT_PIPELINE_MODEL_PARALLEL_RANKS, \
            "Current rank does not have a a next pipeline model parallel rank"
        return _NEXT_PIPELINE_MODEL_PARALLEL_RANKS
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        return [_PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]]


def get_pipeline_model_parallel_prev_rank(index=0):
    """Return the global rank that preceeds the caller in the pipeline (based on input index)"""
    if get_using_layer_unit_test_strategy():
        assert len(_PREV_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "Previous pipeline parallel group is not initialized"
        assert -1 not in _PREV_PIPELINE_MODEL_PARALLEL_RANKS, \
            "Current rank does not have a a previous pipeline model parallel rank"
        return _PREV_PIPELINE_MODEL_PARALLEL_RANKS[index]
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_pipeline_model_parallel_prev_ranks():
    """Return all global rank that preceeds the caller in the pipeline"""
    if get_using_layer_unit_test_strategy():
        assert len(_PREV_PIPELINE_MODEL_PARALLEL_RANKS) != 0, \
            "Previous pipeline parallel group is not initialized"
        assert -1 not in _PREV_PIPELINE_MODEL_PARALLEL_RANKS, \
            "Current rank does not have a a previous pipeline model parallel rank"
        return _PREV_PIPELINE_MODEL_PARALLEL_RANKS
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, \
            "Pipeline parallel group is not initialized"
        rank_in_pipeline = get_pipeline_model_parallel_rank()
        world_size = get_pipeline_model_parallel_world_size()
        return [_PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def get_tensor_and_expert_parallel_rank():
    """Return my rank for the tensor and expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_tensor_and_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    # _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUPS
    _MODEL_PARALLEL_GROUPS = []
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUPS
    _PIPELINE_MODEL_PARALLEL_GROUPS = []
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None
    # specific to LUTS Strategy
    global _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS
    _PREV_COMPONENT_PIPELINE_CONNECTOR_GROUPS = []
    global _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS
    _NEXT_COMPONENT_PIPELINE_CONNECTOR_GROUPS = []
    global _PREV_COMPONENT_PIPELINE_GLOBAL_RANKS
    _PREV_COMPONENT_PIPELINE_GLOBAL_RANKS = []
    global _NEXT_COMPONENT_PIPELINE_GLOBAL_RANKS
    _NEXT_COMPONENT_PIPELINE_GLOBAL_RANKS = []
    global _PREV_PIPELINE_MODEL_PARALLEL_RANKS
    _PREV_PIPELINE_MODEL_PARALLEL_RANKS = []
    global _NEXT_PIPELINE_MODEL_PARALLEL_RANKS
    _NEXT_PIPELINE_MODEL_PARALLEL_RANKS = []
    global _FIRST_PIPELINE_MODEL_PARALLEL_RANKS
    _FIRST_PIPELINE_MODEL_PARALLEL_RANKS = []
    global _LAST_PIPELINE_MODEL_PARALLEL_RANKS
    _LAST_PIPELINE_MODEL_PARALLEL_RANKS = []
    global _NUM_COMPONENT_LAYERS
    _NUM_COMPONENT_LAYERS = None
    global _USING_LAYER_UNIT_TEST_STRATEGY
    _USING_LAYER_UNIT_TEST_STRATEGY = None
    global _FIFO_RATIO
    _FIFO_RATIO = None
    global _IS_RANK_IN_FIRST_COMPONENT
    _IS_RANK_IN_FIRST_COMPONENT = False
    global _IS_RANK_IN_LAST_COMPONENT
    _IS_RANK_IN_LAST_COMPONENT = False

