import importlib.util
import pathlib
import sys
import types

import pytest

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

prompt_parser_path = repo_root / "AUTOMATIC1111" / "stable-diffusion-webui" / "modules" / "prompt_parser.py"
spec = importlib.util.spec_from_file_location("modules.prompt_parser", prompt_parser_path)
prompt_parser = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(prompt_parser)
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name == "lark":
        pytest.skip("prompt_parser requires lark", allow_module_level=True)
    raise

modules_pkg = types.ModuleType("modules")
script_callbacks_mod = types.ModuleType("modules.script_callbacks")
script_callbacks_mod.on_script_unloaded = lambda _callback: None


class _DummyModelWrapCfg:
    def __init__(self):
        self.combine_denoised = lambda *args, **kwargs: None


def _create_sampler(_name, _model):
    return types.SimpleNamespace(model_wrap_cfg=_DummyModelWrapCfg())


sd_samplers_mod = types.ModuleType("modules.sd_samplers")
sd_samplers_mod.create_sampler = _create_sampler

shared_mod = types.ModuleType("modules.shared")
shared_mod.state = types.SimpleNamespace(sampling_step=0)

modules_pkg.prompt_parser = prompt_parser
modules_pkg.script_callbacks = script_callbacks_mod
modules_pkg.sd_samplers = sd_samplers_mod
modules_pkg.shared = shared_mod

sys.modules.setdefault("modules", modules_pkg)
sys.modules["modules.prompt_parser"] = prompt_parser
sys.modules["modules.script_callbacks"] = script_callbacks_mod
sys.modules["modules.sd_samplers"] = sd_samplers_mod
sys.modules["modules.shared"] = shared_mod

try:  # pragma: no cover - torch is optional for local testing
    import torch
except Exception:  # pragma: no cover - torch is optional in CI
    torch = None

from lib_neutral_prompt import global_state, prompt_parser_hijack
from lib_neutral_prompt.cfg_denoiser_hijack import (
    BatchCondAdapter,
    ReindexedCondInfo,
    combine_denoised_hijack,
)


def _make_multicond(weights):
    schedules = [prompt_parser.ScheduledPromptConditioning(5, object())]
    parts = [prompt_parser.ComposableScheduledPromptConditioning(schedules, w) for w in weights]
    return prompt_parser.MulticondLearnedConditioning(shape=(len(weights),), batch=[parts])


def test_batch_cond_adapter_handles_multicond_learned_conditioning():
    multicond = _make_multicond([1.5, 0.25])
    adapter = BatchCondAdapter(multicond)

    assert len(adapter.normalized_batch) == 1
    assert [info.weight for info in adapter.normalized_batch[0]] == [1.5, 0.25]

    reindexed_prompt = [
        ReindexedCondInfo(info, new_index=i + 3, weight=info.weight * 2.0)
        for i, info in enumerate(adapter.normalized_batch[0])
    ]

    converted = adapter.convert_batch([reindexed_prompt])

    assert isinstance(converted, prompt_parser.MulticondLearnedConditioning)
    assert converted is not multicond
    assert [part.weight for part in converted.batch[0]] == [3.0, 0.5]
    assert [part.weight for part in multicond.batch[0]] == [1.5, 0.25]
    assert converted.batch[0][0] is not multicond.batch[0][0]
    assert converted.batch[0][0].schedules is multicond.batch[0][0].schedules


def test_batch_cond_adapter_preserves_tuple_batches():
    tuple_batch = [[(0, 0.75)], [(1, 1.25)]]
    adapter = BatchCondAdapter(tuple_batch)

    reindexed = [
        [ReindexedCondInfo(adapter.normalized_batch[0][0], new_index=4, weight=1.5)],
        [ReindexedCondInfo(adapter.normalized_batch[1][0], new_index=7, weight=2.0)],
    ]

    converted = adapter.convert_batch(reindexed)

    assert converted == [[(4, 1.5)], [(7, 2.0)]]


@pytest.mark.skipif(torch is None, reason="torch required for combine hijack integration test")
def test_combine_denoised_hijack_handles_multicond_batch():
    previous_enabled = global_state.is_enabled
    previous_prompts = global_state.prompt_exprs
    previous_cfg_rescale = global_state.cfg_rescale
    previous_cfg_rescale_factor = global_state.cfg_rescale_factor

    try:
        global_state.is_enabled = True
        prompt = "cat AND dog :0.5"
        global_state.prompt_exprs = prompt_parser_hijack.parse_prompts([prompt])

        schedules = [prompt_parser.ScheduledPromptConditioning(10, torch.zeros((1, 1, 1)))]
        parts = [
            prompt_parser.ComposableScheduledPromptConditioning(schedules, 1.0),
            prompt_parser.ComposableScheduledPromptConditioning(schedules, 0.5),
        ]
        multicond = prompt_parser.MulticondLearnedConditioning(shape=(2,), batch=[parts])

        cond_a = torch.full((1, 1, 1), 0.1)
        cond_b = torch.full((1, 1, 1), 0.2)
        uncond = torch.zeros((1, 1, 1))
        x_out = torch.stack([cond_a, cond_b, uncond], dim=0)
        text_uncond = torch.zeros((1, 1, 1))
        cond_scale = 1.0

        captured = {}

        def original_function(sliced_x_out, converted_batch, text_uncond_arg, cond_scale_arg):
            captured["type"] = type(converted_batch)
            if hasattr(converted_batch, "batch"):
                captured["weights"] = [
                    [part.weight for part in prompt_parts] for prompt_parts in converted_batch.batch
                ]
            captured["shape"] = sliced_x_out.shape
            assert text_uncond_arg is text_uncond
            assert cond_scale_arg == cond_scale
            return sliced_x_out.clone()

        result = combine_denoised_hijack(
            x_out=x_out,
            batch_cond_indices=multicond,
            text_uncond=text_uncond,
            cond_scale=cond_scale,
            original_function=original_function,
        )

        assert captured["type"] is prompt_parser.MulticondLearnedConditioning
        assert captured["weights"] == [[1.0, 0.5]]
        assert captured["shape"] == torch.Size([3, 1, 1, 1])
        assert result.shape == x_out.shape
        assert torch.allclose(result[-1], x_out[-1])
    finally:
        global_state.is_enabled = previous_enabled
        global_state.prompt_exprs = previous_prompts
        global_state.cfg_rescale = previous_cfg_rescale
        global_state.cfg_rescale_factor = previous_cfg_rescale_factor
