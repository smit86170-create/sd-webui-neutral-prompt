from lib_neutral_prompt import hijacker, global_state, neutral_prompt_parser
from modules import prompt_parser, script_callbacks, sd_samplers, shared
from typing import Tuple, List, Any
import dataclasses
import functools
import torch
import sys
import textwrap


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
        self.format = self._detect_format(batch_cond_indices)
        self.composable_cls = (
            getattr(prompt_parser, 'ComposableScheduledPromptConditioning', None)
            if self.format == 'composable'
            else None
        )

        if self.format == 'composable' and self.composable_cls is None:
            raise AttributeError('ComposableScheduledPromptConditioning not available in prompt_parser')

        self.normalized_batch = self._normalize(batch_cond_indices)

    def _detect_format(self, batch_cond_indices):
        for conds in batch_cond_indices or []:
            for cond in conds:
                if isinstance(cond, (tuple, list)):
                    return 'tuple'
                if hasattr(cond, 'weight') and hasattr(cond, 'schedules'):
                    return 'composable'
                raise TypeError('Unsupported conditioning entry type')
        return 'tuple'

    def _normalize(self, batch_cond_indices):
        normalized = []
        running_index = 0
        for conds in batch_cond_indices:
            prompt_infos = []
            for cond in conds:
                if self.format == 'tuple':
                    cond_index, weight = cond
                    prompt_infos.append(NormalizedCondInfo(int(cond_index), float(weight), cond))
                else:
                    prompt_infos.append(NormalizedCondInfo(running_index, float(cond.weight), cond))
                    running_index += 1
            normalized.append(prompt_infos)
        return normalized

    def convert_prompt(self, prompt_infos):
        if self.format == 'tuple':
            return [(info.new_index, info.weight) for info in prompt_infos]

        return [
            self.composable_cls(info.info.original.schedules, info.weight)
            for info in prompt_infos
        ]

    def convert_batch(self, batch_infos):
        return [self.convert_prompt(prompt_infos) for prompt_infos in batch_infos]

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
        args = CombineDenoiseArgs(x_out, uncond[batch_i], cond_infos)
        cond_delta = prompt.accept(CondDeltaVisitor(), args, 0)
        aux_cond_delta = prompt.accept(AuxCondDeltaVisitor(), args, cond_delta, 0)
        cfg_cond = denoised[batch_i] + aux_cond_delta * cond_scale
        denoised[batch_i] = cfg_rescale(cfg_cond, uncond[batch_i] + cond_delta + aux_cond_delta)

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
        args = CombineDenoiseArgs(x_out, uncond[batch_i], cond_infos)
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
        return cfg_cond

    global_state.apply_and_clear_cfg_rescale_override()
    cfg_cond_mean = cfg_cond.mean()
    cfg_rescale_mean = (1 - global_state.cfg_rescale) * cfg_cond_mean + global_state.cfg_rescale * cond.mean()
    cfg_rescale_factor = global_state.cfg_rescale * (cond.std() / cfg_cond.std() - 1) + 1
    return cfg_rescale_mean + (cfg_cond - cfg_cond_mean) * cfg_rescale_factor


@dataclasses.dataclass
class CombineDenoiseArgs:
    x_out: torch.Tensor
    uncond: torch.Tensor
    cond_infos: List[NormalizedCondInfo]


def gather_webui_conds(
    prompt: neutral_prompt_parser.CompositePrompt,
    args: CombineDenoiseArgs,
    index_in: int,
    index_out: int,
) -> Tuple[List[torch.Tensor], List[ReindexedCondInfo]]:
    sliced_x_out = []
    sliced_cond_infos = []

    for child in prompt.children:
        if child.conciliation is None:
            if isinstance(child, neutral_prompt_parser.LeafPrompt):
                child_info = args.cond_infos[index_in]
                child_x_out = args.x_out[child_info.x_out_index]
            else:
                child_x_out = child.accept(CondDeltaVisitor(), args, index_in)
                child_x_out += child.accept(AuxCondDeltaVisitor(), args, child_x_out, index_in)
                child_x_out += args.uncond
            index_offset = index_out + len(sliced_x_out)
            sliced_x_out.append(child_x_out)
            sliced_cond_infos.append(ReindexedCondInfo(args.cond_infos[index_in], index_offset, child.weight))

        index_in += child.accept(neutral_prompt_parser.FlatSizeVisitor())

    return sliced_x_out, sliced_cond_infos


class CondDeltaVisitor:
    def visit_leaf_prompt(
        self,
        that: neutral_prompt_parser.LeafPrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_info = args.cond_infos[index]
        if that.weight != cond_info.weight:
            console_warn(f'''
                An unexpected noise weight was encountered at prompt #{index}
                Expected :{that.weight}, but got :{cond_info.weight}
                This is likely due to another extension also monkey patching the webui `combine_denoised` function
                Please open a bug report here so that the conflict can be resolved:
                https://github.com/ljleb/sd-webui-neutral-prompt/issues
            ''')

        return args.x_out[cond_info.x_out_index] - args.uncond

    def visit_composite_prompt(
        self,
        that: neutral_prompt_parser.CompositePrompt,
        args: CombineDenoiseArgs,
        index: int,
    ) -> torch.Tensor:
        cond_delta = torch.zeros_like(args.x_out[0])

        for child in that.children:
            if child.conciliation is None:
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index)
                cond_delta += child.weight * child_cond_delta

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

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

        for child in that.children:
            if child.conciliation is not None:
                child_cond_delta = child.accept(CondDeltaVisitor(), args, index)
                child_cond_delta += child.accept(AuxCondDeltaVisitor(), args, child_cond_delta, index)

                if child.conciliation == neutral_prompt_parser.ConciliationStrategy.PERPENDICULAR:
                    aux_cond_delta += child.weight * get_perpendicular_component(cond_delta, child_cond_delta)
                elif child.conciliation == neutral_prompt_parser.ConciliationStrategy.SALIENCE_MASK:
                    salient_cond_deltas.append((child_cond_delta, child.weight))
                elif child.conciliation == neutral_prompt_parser.ConciliationStrategy.SEMANTIC_GUIDANCE:
                    aux_cond_delta += child.weight * filter_abs_top_k(child_cond_delta, 0.05)

            index += child.accept(neutral_prompt_parser.FlatSizeVisitor())

        aux_cond_delta += salient_blend(cond_delta, salient_cond_deltas)
        return aux_cond_delta


def get_perpendicular_component(normal: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if (normal == 0).all():
        if shared.state.sampling_step <= 0:
            warn_projection_not_found()

        return vector

    return vector - normal * torch.sum(normal * vector) / torch.norm(normal) ** 2


def salient_blend(normal: torch.Tensor, vectors: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    """
        Blends the `normal` tensor with `vectors` in salient regions, weighting contributions by their weights.
        Salience maps are calculated to identify regions of interest.
        The blended result combines `normal` and vector information in salient regions.
    """

    salience_maps = [get_salience(normal)] + [get_salience(vector) for vector, _ in vectors]
    mask = torch.argmax(torch.stack(salience_maps, dim=0), dim=0)

    result = torch.zeros_like(normal)
    for mask_i, (vector, weight) in enumerate(vectors, start=1):
        vector_mask = (mask == mask_i).float()
        result += weight * vector_mask * (vector - normal)

    return result


def get_salience(vector: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.abs(vector).flatten(), dim=0).reshape_as(vector)


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
        combine_denoised_hijack,
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
