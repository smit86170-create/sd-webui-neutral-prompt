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

from lib_neutral_prompt.cfg_denoiser_hijack import BatchCondAdapter, ReindexedCondInfo


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
