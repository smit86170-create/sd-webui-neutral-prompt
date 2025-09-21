import importlib.util
import pathlib
import sys
import types

import pytest

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

prompt_parser_path = (
    repo_root / "AUTOMATIC1111" / "stable-diffusion-webui" / "modules" / "prompt_parser.py"
)
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

from lib_neutral_prompt import neutral_prompt_parser


def test_alignment_blend_keyword_parsing():
    expr = neutral_prompt_parser.parse_root('foo AND_ALIGN_4_8 [bar AND baz]')
    assert isinstance(expr.children[1], neutral_prompt_parser.CompositePrompt)
    child = expr.children[1]
    assert child.conciliation == neutral_prompt_parser.ConciliationStrategy.ALIGNMENT_BLEND
    assert child.conciliation_args == (4, 8)


def test_alignment_mask_keyword_parsing():
    expr = neutral_prompt_parser.parse_root('foo AND_MASK_ALIGN_7_5 [bar AND baz]')
    child = expr.children[1]
    assert child.conciliation == neutral_prompt_parser.ConciliationStrategy.ALIGNMENT_MASK
    assert child.conciliation_args == (7, 5)


def test_affine_transform_parse_leaf():
    expr = neutral_prompt_parser.parse_root('ROTATE[0.25] test prompt')
    leaf = expr.children[0]
    assert isinstance(leaf, neutral_prompt_parser.LeafPrompt)
    assert leaf.local_transform is not None
    assert len(leaf.local_transform) == 2
    assert len(leaf.local_transform[0]) == 3


def test_make_alignment_keyword_helpers():
    assert neutral_prompt_parser.make_alignment_keyword(3, 9) == 'AND_ALIGN_3_9'
    assert neutral_prompt_parser.make_alignment_mask_keyword(2, 5) == 'AND_MASK_ALIGN_2_5'


def test_scheduled_prompt_with_stray_and_sequence():
    expr = neutral_prompt_parser.parse_root("[foo:bar:0.5]:10 AND_PERP AND baz")

    assert len(expr.children) == 3

    first = expr.children[0]
    assert isinstance(first, neutral_prompt_parser.LeafPrompt)
    assert first.prompt == '[foo:bar:0.5]'
    assert first.weight == pytest.approx(10.0)

    second = expr.children[1]
    assert isinstance(second, neutral_prompt_parser.LeafPrompt)
    assert second.conciliation == neutral_prompt_parser.ConciliationStrategy.PERPENDICULAR
    assert second.prompt == ''
    assert second.weight == pytest.approx(1.0)

    third = expr.children[2]
    assert isinstance(third, neutral_prompt_parser.LeafPrompt)
    assert third.conciliation is None
    assert third.prompt == ' baz'
    assert third.weight == pytest.approx(1.0)


def test_brace_group_weight_and_alignment_mask_chain():
    expr = neutral_prompt_parser.parse_root("{a,b}:1.2 AND_MASK_ALIGN_3_4 [foo AND bar]")

    assert len(expr.children) == 2

    first = expr.children[0]
    assert isinstance(first, neutral_prompt_parser.LeafPrompt)
    assert first.prompt == '{a,b}'
    assert first.weight == pytest.approx(1.2)

    second = expr.children[1]
    assert isinstance(second, neutral_prompt_parser.CompositePrompt)
    assert second.conciliation == neutral_prompt_parser.ConciliationStrategy.ALIGNMENT_MASK
    assert second.conciliation_args == (3, 4)
    prompts = [child.prompt for child in second.children]
    assert [prompt.strip() for prompt in prompts] == ['foo', 'bar']


def test_salt_block_accepts_comma_separated_weighted_groups():
    prompt = (
        "2girls, harime nui, matoi ryuuko, selfie, view from above, kill la kill style, anime\n"
        "AND_SALT [\n"
        "{matoi ryuuko, huge breasts} :1.5,\n"
        "{harime nui, flat chest} :0.8\n"
        "]"
    )

    expr = neutral_prompt_parser.parse_root(prompt)

    assert len(expr.children) == 2

    salt_group = expr.children[1]
    assert isinstance(salt_group, neutral_prompt_parser.CompositePrompt)
    assert len(salt_group.children) == 2

    first, second = salt_group.children
    assert isinstance(first, neutral_prompt_parser.LeafPrompt)
    assert isinstance(second, neutral_prompt_parser.LeafPrompt)
    assert first.prompt.strip() == '{matoi ryuuko, huge breasts}'
    assert second.prompt.strip() == '{harime nui, flat chest}'
    assert first.weight == pytest.approx(1.5)
    assert second.weight == pytest.approx(0.8)

try:
    import torch
except Exception:  # pragma: no cover - torch is optional in tests
    torch = None


@pytest.mark.skipif(torch is None, reason='torch required for local transform tests')
def test_apply_local_transform_identity():
    from lib_neutral_prompt import cfg_denoiser_hijack

    cond_delta = torch.zeros((4, 8, 8))
    uncond = torch.zeros_like(cond_delta)
    transform = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    result = cfg_denoiser_hijack.apply_local_transform_to_delta(cond_delta, uncond, transform, 1.0)
    assert torch.allclose(result, cond_delta)


@pytest.mark.skipif(torch is None, reason='torch required for life tests')
def test_life_preserves_shape_for_multi_channel_board():
    from lib_neutral_prompt import cfg_denoiser_hijack

    board = torch.zeros((3, 4, 4))
    board[0, 1, 1] = 1.0

    result = cfg_denoiser_hijack.life(
        board,
        iterations=1,
        birth_threshold=0.1,
        survive_min=0.0,
        survive_max=1.0,
    )

    assert result.shape == board.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


@pytest.mark.skipif(torch is None, reason='torch required for life tests')
def test_life_accepts_2d_masks():
    from lib_neutral_prompt import cfg_denoiser_hijack

    board = torch.zeros((4, 4))
    result = cfg_denoiser_hijack.life(board, iterations=1)

    assert result.shape == board.shape
