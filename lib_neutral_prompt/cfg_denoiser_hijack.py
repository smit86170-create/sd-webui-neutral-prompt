from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import prompt_parser, script_callbacks, sd_samplers, shared
from typing import Tuple, List, Any, Optional, Dict
import dataclasses
import functools
import torch
import torch.nn.functional as F
import sys
import textwrap
import copy


@dataclasses.dataclass(frozen=True)
class NormalizedCondInfo:
    x_out_index: int
    weight: float
    original: Any


@dataclasses.dataclass(frozen=True)
class ReindexedCondInfo:
    info: NormalizedCondInfo
    new_index: int
    weight: float


class BatchCondAdapter:
    def __init__(self, batch_cond_indices):
        self.original_batch = batch_cond_indices
        self.multicond_cls = getattr(prompt_parser, 'MulticondLearnedConditioning', None)
        self.composable_cls = getattr(prompt_parser, 'ComposableScheduledPromptConditioning', None)
        self.container_type, batch_view = self._extract_container(batch_cond_indices)
        self.entry_type = self._detect_entry_type(batch_view)

        if self.entry_type == 'composable' and self.composable_cls is None:
            raise AttributeError('ComposableScheduledPromptConditioning not available in prompt_parser')

        self.normalized_batch = self._normalize(batch_view)

    def _extract_container(self, batch_cond_indices):
        if batch_cond_indices is None:
            return 'list', []

        if self.multicond_cls is not None and isinstance(batch_cond_indices, self.multicond_cls):
            return 'multicond', batch_cond_indices.batch

        return 'list', batch_cond_indices

    def _detect_entry_type(self, batch_view):
        for conds in batch_view or []:
            for cond in conds:
                if isinstance(cond, (tuple, list)):
                    return 'tuple'
                if hasattr(cond, 'weight') and hasattr(cond, 'schedules'):
                    return 'composable'
                raise TypeError('Unsupported conditioning entry type')
        return 'tuple'

    def _normalize(self, batch_view):
        normalized = []
        running_index = 0
        for conds in batch_view or []:
            prompt_infos = []
            for cond in conds:
                if self.entry_type == 'tuple':
                    cond_index, weight = cond
                    prompt_infos.append(NormalizedCondInfo(int(cond_index), float(weight), cond))
                else:
                    prompt_infos.append(NormalizedCondInfo(running_index, float(cond.weight), cond))
                    running_index += 1
            normalized.append(prompt_infos)
        return normalized

    def convert_prompt(self, prompt_infos):
        if self.entry_type == 'tuple':
            return [(info.new_index, info.weight) for info in prompt_infos]

        converted = []
        for info in prompt_infos:
            original = info.info.original
            cloned = copy.copy(original)
            cloned.weight = info.weight
            converted.append(cloned)
        return converted

    def convert_batch(self, batch_infos):
        converted = [self.convert_prompt(prompt_infos) for prompt_infos in batch_infos]

        if self.container_type == 'multicond':
            if self.multicond_cls is None:
                raise AttributeError('MulticondLearnedConditioning not available in prompt_parser')

            cloned = copy.copy(self.original_batch)
            cloned.batch = converted
            return cloned

        return converted

def _select_uncond(uncond: torch.Tensor, index: int) -> torch.Tensor:
    if uncond.ndim == 0:
        return uncond

    uncond_count = uncond.shape[0]
    if uncond_count == 0:
        raise ValueError('Unconditional batch is empty')

    if uncond_count == 1:
        return uncond[0]

    if index < uncond_count:
        return uncond[index]

    return uncond[index % uncond_count]


def combine_denoised_hijack(
    x_out: torch.Tensor,
    batch_cond_indices,
    text_uncond: torch.Tensor,
    cond_scale: float,
    original_function,
) -> torch.Tensor:
    if not global_state.is_enabled:
        return original_function(x_out, batch_cond_indices, text_uncond, cond_scale)

    adapter = BatchCondAdapter(batch_cond_indices)
    denoised = get_webui_denoised(x_out, adapter, text_uncond, cond_scale, original_function)
    uncond = x_out[-text_uncond.shape[0]:]

    for batch_i, (prompt, cond_infos) in enumerate(zip(global_state.prompt_exprs, adapter.normalized_batch)):
        uncond_tensor = _select_uncond(uncond, batch_i)
        args = CombineDenoiseArgs(x_out, uncond_tensor, cond_infos)
        cond_delta = prompt.accept(CondDeltaVisitor(), args, 0)
        aux_cond_delta = prompt.accept(AuxCondDeltaVisitor(), args, cond_delta, 0)
        cfg_cond = denoised[batch_i] + aux_cond_delta * cond_scale
        denoised[batch_i] = cfg_rescale(cfg_cond, uncond_tensor + cond_delta + aux_cond_delta)

    return denoised


def get_webui_denoised(
    x_out: torch.Tensor,
    adapter: BatchCondAdapter,
    text_uncond: torch.Tensor,
    cond_scale: float,
    original_function,
):
    uncond = x_out[-text_uncond.shape[0]:]
    sliced_batch_x_out = []
    sliced_batch_cond_infos = []

    for batch_i, (prompt, cond_infos) in enumerate(zip(global_state.prompt_exprs, adapter.normalized_batch)):
        uncond_tensor = _select_uncond(uncond, batch_i)
        args = CombineDenoiseArgs(x_out, uncond_tensor, cond_infos)
        sliced_x_out, reindexed_infos = gather_webui_conds(prompt, args, 0, len(sliced_batch_x_out))
        if reindexed_infos:
            sliced_batch_cond_infos.append(reindexed_infos)
        sliced_batch_x_out.extend(sliced_x_out)

    sliced_batch_x_out += list(uncond)
    sliced_batch_x_out = torch.stack(sliced_batch_x_out, dim=0)
    converted_cond_infos = adapter.convert_batch(sliced_batch_cond_infos)
    return original_function(sliced_batch_x_out, converted_cond_infos, text_uncond, cond_scale)


def cfg_rescale(cfg_cond, cond):
    if global_state.cfg_rescale == 0:
        global_state.cfg_rescale_factor = 1.0
        return cfg_cond

    global_state.apply_and_clear_cfg_rescale_override()
    cfg_cond_mean = cfg_cond.mean()
    cfg_rescale_mean = (1 - global_state.cfg_rescale) * cfg_cond_mean + global_state.cfg_rescale * cond.mean()
    cfg_cond_std = cfg_cond.std()
    if torch.isclose(cfg_cond_std, torch.zeros((), device=cfg_cond_std.device, dtype=cfg_cond_std.dtype)):
        cfg_rescale_factor = torch.ones_like(cfg_cond_std)
    else:
        cfg_rescale_factor = global_state.cfg_rescale * (cond.std() / cfg_cond_std - 1) + 1

    factor_value = float(cfg_rescale_factor.item()) if hasattr(cfg_rescale_factor, 'item') else float(cfg_rescale_factor)
    global_state.cfg_rescale_factor = factor_value
    return cfg_rescale_mean + (cfg_cond - cfg_cond_mean) * cfg_rescale_factor


@dataclasses.dataclass
class CombineDenoiseArgs:
    x_out: torch.Tensor
    uncond: torch.Tensor
    cond_infos: List[NormalizedCondInfo]
    leaf_cond_cache: Dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)
    leaf_delta_cache: Dict[int, torch.Tensor] = dataclasses.field(default_factory=dict)


def gather_webui_conds(
    prompt: neutral_prompt_parser.CompositePrompt,
    args: CombineDenoiseArgs,
    index_in: int,
    index_out: int,
) -> Tuple[List[torch.Tensor], List[ReindexedCondInfo]]:
    sliced_x_out = []
    sliced_cond_infos = []

    for child in prompt.children:
        child_flat_size = child.accept(neutral_prompt_parser.FlatSizeVisitor())
        if child.conciliation is None:
            if isinstance(child, neutral_prompt_parser.LeafPrompt):
                child_x_out, _child_delta, child_info = evaluate_leaf_prompt(child, args, index_in)
            else:
                child_info = args.cond_infos[index_in]
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index_in)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index_in)
                child_x_out = args.uncond + child_cond_delta
            index_offset = index_out + len(sliced_x_out)
            sliced_x_out.append(child_x_out)
            sliced_cond_infos.append(ReindexedCondInfo(child_info, index_offset, child.weight))

        index_in += child_flat_size

    return sliced_x_out, sliced_cond_infos


def evaluate_leaf_prompt(
    leaf: neutral_prompt_parser.LeafPrompt,
    args: CombineDenoiseArgs,
    index: int,
) -> Tuple[torch.Tensor, torch.Tensor, NormalizedCondInfo]:
    if index in args.leaf_cond_cache:
        return args.leaf_cond_cache[index], args.leaf_delta_cache[index], args.cond_infos[index]

    cond_info = args.cond_infos[index]
    cond_tensor = args.x_out[cond_info.x_out_index]

    if leaf.weight != cond_info.weight:
        console_warn(f'''
            An unexpected noise weight was encountered at prompt #{index}
            Expected :{leaf.weight}, but got :{cond_info.weight}
            This is likely due to another extension also monkey patching the webui `combine_denoised` function
            Please open a bug report here so that the conflict can be resolved:
            https://github.com/ljleb/sd-webui-neutral-prompt/issues
        ''')

    cond_delta = cond_tensor - args.uncond
    cond_delta = apply_local_transform_to_delta(cond_delta, args.uncond, leaf.local_transform, leaf.weight)
    cond_tensor = args.uncond + cond_delta

    args.leaf_cond_cache[index] = cond_tensor
    args.leaf_delta_cache[index] = cond_delta

    return cond_tensor, cond_delta, cond_info


def apply_local_transform_to_delta(
    cond_delta: torch.Tensor,
    uncond: torch.Tensor,
    transform: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    weight: float,
) -> torch.Tensor:
    if transform is None:
        return cond_delta

    cond = cond_delta + uncond
    transform_tensor = torch.as_tensor(transform, dtype=cond.dtype, device=cond.device)
    mask = create_cosine_feathered_mask(cond.shape[-2:], weight).to(cond.device, cond.dtype)
    tensor_with_mask = torch.cat([cond, mask.unsqueeze(0)], dim=0)
    transformed = apply_affine_transform(tensor_with_mask, transform_tensor)
    transformed_cond = transformed[:-1]
    transformed_mask = transformed[-1].clamp_(0.0, 1.0).unsqueeze(0)
    transformed_delta = transformed_cond - uncond
    return transformed_mask * transformed_delta + (1 - transformed_mask) * cond_delta


class CondDeltaVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        _, cond_delta, _ = evaluate_leaf_prompt(that, args, index)
        return cond_delta

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_delta = torch.zeros_like(args.x_out[0])

        for child in that.children:
            child_flat_size = child.accept(neutral_prompt_parser.FlatSizeVisitor())
            if child.conciliation is None:
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index)
                cond_delta += child.weight * child_cond_delta

            index += child_flat_size

        cond_delta = apply_local_transform_to_delta(cond_delta, args.uncond, that.local_transform, that.weight)
        return cond_delta


class AuxCondDeltaVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombineDenoiseArgs,
        cond_delta: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        return torch.zeros_like(args.x_out[0])

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        cond_delta: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        aux_cond_delta = torch.zeros_like(args.x_out[0])
        salient_cond_deltas = []
        alignment_cond_deltas: List[Tuple[torch.Tensor, float, int, int]] = []
        alignment_mask_cond_deltas: List[Tuple[torch.Tensor, float, int, int]] = []

        for child in that.children:
            child_flat_size = child.accept(neutral_prompt_parser.FlatSizeVisitor())
            if child.conciliation is not None:
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index)

                strategy = child.conciliation
                if strategy == neutral_prompt_parser.ConciliationStrategy.PERPENDICULAR:
                    aux_cond_delta += child.weight * get_perpendicular_component(cond_delta, child_cond_delta)
                elif strategy == neutral_prompt_parser.ConciliationStrategy.SALIENCE_MASK:
                    salient_cond_deltas.append((child_cond_delta, child.weight))
                elif strategy == neutral_prompt_parser.ConciliationStrategy.SEMANTIC_GUIDANCE:
                    aux_cond_delta += child.weight * filter_abs_top_k(child_cond_delta, 0.05)
                elif strategy == neutral_prompt_parser.ConciliationStrategy.ALIGNMENT_BLEND:
                    detail, structure = child.conciliation_args or (4, 8)
                    alignment_cond_deltas.append((child_cond_delta, child.weight, detail, structure))
                elif strategy == neutral_prompt_parser.ConciliationStrategy.ALIGNMENT_MASK:
                    detail, structure = child.conciliation_args or (4, 8)
                    alignment_mask_cond_deltas.append((child_cond_delta, child.weight, detail, structure))

            index += child_flat_size

        if salient_cond_deltas:
            aux_cond_delta += salient_blend(cond_delta, salient_cond_deltas)
        if alignment_cond_deltas:
            aux_cond_delta += alignment_blend(cond_delta, alignment_cond_deltas)
        if alignment_mask_cond_deltas:
            aux_cond_delta += alignment_mask_blend(cond_delta, alignment_mask_cond_deltas)

        aux_cond_delta = apply_local_transform_to_delta(aux_cond_delta, args.uncond, that.local_transform, that.weight)
        return aux_cond_delta


def create_cosine_feathered_mask(size: Tuple[int, int], weight: float) -> torch.Tensor:
    height, width = size
    y = torch.linspace(-1.0, 1.0, steps=height)
    x = torch.linspace(-1.0, 1.0, steps=width)
    yy, xx = torch.meshgrid(y, x)
    dist = torch.sqrt(xx ** 2 + yy ** 2)
    mask = 0.5 * (1 + torch.cos(torch.pi * dist))
    mask = torch.where(dist <= 1, mask, torch.zeros_like(mask))
    return mask.float() * abs(weight)


def apply_affine_transform(tensor: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    affine = affine.to(dtype=tensor.dtype, device=tensor.device)
    aspect_ratio = tensor.shape[-2] / tensor.shape[-1]
    adjusted = affine.clone()
    adjusted[0, 1] *= aspect_ratio
    adjusted[1, 0] /= aspect_ratio

    grid = F.affine_grid(adjusted.unsqueeze(0), tensor.unsqueeze(0).size(), align_corners=False)
    transformed_tensors = F.grid_sample(
        tensor.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    return transformed_tensors.squeeze(0)


def normalize_similarity_map(tensor: torch.Tensor) -> torch.Tensor:
    max_value = tensor.max()
    if max_value <= 0:
        return torch.zeros_like(tensor)
    return tensor / max_value


def compute_subregion_similarity_map(
    child_vector: torch.Tensor,
    parent_vector: torch.Tensor,
    *,
    region_size: int = 2,
) -> torch.Tensor:
    region_size = max(int(region_size), 2)
    channels, height, width = child_vector.shape
    parent = parent_vector.unsqueeze(0)
    child = child_vector.unsqueeze(0)

    region_radius = region_size // 2
    if region_size % 2 == 1:
        pad_size = (region_radius,) * 4
    else:
        pad_size = (region_radius - 1, region_radius, region_radius - 1, region_radius)

    parent_regions = F.pad(parent, pad_size, mode='constant', value=0)
    child_regions = F.pad(child, pad_size, mode='constant', value=0)
    unfold = torch.nn.Unfold(kernel_size=region_size)
    parent_regions = unfold(parent_regions)
    child_regions = unfold(child_regions)

    parent_regions = parent_regions.view(1, channels, region_size ** 2, height * width)
    parent_regions = parent_regions.permute(3, 1, 2, 0).view(height * width, channels, region_size, region_size)
    child_regions = child_regions.view(1, channels, region_size ** 2, height * width)
    child_regions = child_regions.permute(3, 1, 2, 0).view(height * width, channels, region_size, region_size)

    subregion_unfold = torch.nn.Unfold(kernel_size=2)
    parent_subregions = subregion_unfold(parent_regions).view(height * width, channels, 4, (region_size - 1) ** 2)
    child_subregions = subregion_unfold(child_regions).view(height * width, channels, 4, (region_size - 1) ** 2)

    parent_subregions = F.normalize(parent_subregions, p=2, dim=2)
    child_subregions = F.normalize(child_subregions, p=2, dim=2)
    similarity = (parent_subregions * child_subregions).sum(dim=2)

    return similarity.mean(dim=2).permute(1, 0).view(channels, height, width)


def alignment_blend(parent: torch.Tensor, children: List[Tuple[torch.Tensor, float, int, int]]) -> torch.Tensor:
    if not children:
        return torch.zeros_like(parent)

    result = torch.zeros_like(parent)
    for child, weight, detail_size, structure_size in children:
        detail_alignment = normalize_similarity_map(
            compute_subregion_similarity_map(child, parent, region_size=max(int(detail_size), 2))
        )
        structure_alignment = normalize_similarity_map(
            compute_subregion_similarity_map(child, parent, region_size=max(int(structure_size), 2))
        )

        alignment_weight = torch.clamp(structure_alignment - detail_alignment, min=0.0, max=1.0)
        result += weight * alignment_weight * (child - parent)

    return result


def alignment_mask_blend(parent: torch.Tensor, children: List[Tuple[torch.Tensor, float, int, int]]) -> torch.Tensor:
    if not children:
        return torch.zeros_like(parent)

    result = torch.zeros_like(parent)
    for child, weight, detail_size, structure_size in children:
        detail_alignment = normalize_similarity_map(
            compute_subregion_similarity_map(child, parent, region_size=max(int(detail_size), 2))
        )
        structure_alignment = normalize_similarity_map(
            compute_subregion_similarity_map(child, parent, region_size=max(int(structure_size), 2))
        )

        alignment_mask = (structure_alignment > detail_alignment).to(child.dtype)
        result += weight * alignment_mask * (child - parent)

    return result

def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()

        return vector

    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def life(
    board: torch.Tensor,
    *,
    iterations: int = 1,
    birth_threshold: float = 0.5,
    survive_min: float = 0.3,
    survive_max: float = 0.8,
) -> torch.Tensor:
    if iterations <= 0:
        return board

    if board.ndim not in (2, 3):
        return board

    squeeze_channel_dim = board.ndim == 2
    if squeeze_channel_dim:
        board = board.unsqueeze(0)

    channels = board.shape[0]
    if channels == 0:
        return board.squeeze(0) if squeeze_channel_dim else board

    kernel = torch.ones((channels, 1, 3, 3), dtype=board.dtype, device=board.device)
    state = board.unsqueeze(0)

    for _ in range(iterations):
        neighbors = F.conv2d(state, kernel, padding=1, groups=channels)
        births = (neighbors >= birth_threshold * 9).to(board.dtype)
        survive = ((neighbors >= survive_min * 9) & (neighbors <= survive_max * 9)).to(board.dtype)
        state = torch.clamp(births + survive * state, 0.0, 1.0)

    result = state.squeeze(0)
    if squeeze_channel_dim:
        result = result.squeeze(0)
    return result


def salient_blend(normal: torch.Tensor, vectors: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    if not vectors:
        return torch.zeros_like(normal)

    salience_maps = [get_salience(normal)] + [get_salience(vector, emphasis=20.0) for vector, _ in vectors]
    mask = torch.argmax(torch.stack(salience_maps, dim=0), dim=0)

    result = torch.zeros_like(normal)
    for mask_i, (vector, weight) in enumerate(vectors, start=1):
        vector_mask = (mask == mask_i).to(vector.dtype)
        if torch.count_nonzero(vector_mask) == 0:
            continue

        vector_mask = life(
            vector_mask,
            iterations=2,
            birth_threshold=0.6,
            survive_min=0.4,
            survive_max=0.85,
        )
        vector_mask = F.avg_pool2d(vector_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        vector_mask = torch.clamp(vector_mask, 0.0, 1.0)
        result += weight * vector_mask * (vector - normal)

    return result


def get_salience(vector: torch.Tensor, *, emphasis: float = 1.0) -> torch.Tensor:
    flattened = torch.abs(vector).flatten()
    if flattened.numel() == 0:
        return torch.zeros_like(vector)

    weights = torch.softmax(emphasis * flattened, dim=0)
    return weights.reshape_as(vector)


def filter_abs_top_k(vector: torch.Tensor, k_ratio: float) -> torch.Tensor:
    k = int(torch.numel(vector) * (1 - k_ratio))
    top_k, _ = torch.kthvalue(torch.abs(torch.flatten(vector)), k)
    return vector * (torch.abs(vector) >= top_k).to(vector.dtype)


sd_samplers_hijacker = hijacker.ModuleHijacker.install_or_get(
    module=sd_samplers,
    hijacker_attribute='__neutral_prompt_hijacker',
    on_uninstall=script_callbacks.on_script_unloaded,
)


@sd_samplers_hijacker.hijack('create_sampler')
def create_sampler_hijack(name: str, model, original_function):
    sampler = original_function(name, model)
    if not hasattr(sampler, 'model_wrap_cfg') or not hasattr(sampler.model_wrap_cfg, 'combine_denoised'):
        if global_state.is_enabled:
            warn_unsupported_sampler()

        return sampler

    sampler.model_wrap_cfg.combine_denoised = functools.partial(
@@ -233,25 +577,29 @@ def create_sampler_hijack(name: str, model, original_function):
        original_function=sampler.model_wrap_cfg.combine_denoised
    )
    return sampler


def warn_unsupported_sampler():
    console_warn('''
        Neutral prompt relies on composition via AND, which the webui does not support when using any of the DDIM, PLMS and UniPC samplers
        The sampler will NOT be patched
        Falling back on original sampler implementation...
    ''')


def warn_projection_not_found():
    console_warn('''
        Could not find a projection for one or more AND_PERP prompts
        These prompts will NOT be made perpendicular
    ''')


def console_warn(message):
    if not global_state.verbose:
        return

    print(f'\n[sd-webui-neutral-prompt extension]{textwrap.dedent(message)}', file=sys.stderr)


def get_last_cfg_rescale_factor() -> Optional[float]:
    return getattr(global_state, 'cfg_rescale_factor', None)