import pytest

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
