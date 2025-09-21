import abc
import dataclasses
import math
import re
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple

import pathlib
import sys

import lark

from modules import prompt_parser as webui_prompt_parser

class PromptKeyword(Enum):
    AND = 'AND'
    AND_PERP = 'AND_PERP'
    AND_SALT = 'AND_SALT'
    AND_TOPK = 'AND_TOPK'
    AND_ALIGN = 'AND_ALIGN'
    AND_MASK_ALIGN = 'AND_MASK_ALIGN'


class ConciliationStrategy(Enum):
    PERPENDICULAR = PromptKeyword.AND_PERP.value
    SALIENCE_MASK = PromptKeyword.AND_SALT.value
    SEMANTIC_GUIDANCE = PromptKeyword.AND_TOPK.value
    ALIGNMENT_BLEND = PromptKeyword.AND_ALIGN.value
    ALIGNMENT_MASK = PromptKeyword.AND_MASK_ALIGN.value


prompt_keywords = [
    PromptKeyword.AND.value,
    PromptKeyword.AND_PERP.value,
    PromptKeyword.AND_SALT.value,
    PromptKeyword.AND_TOPK.value,
]

alignment_keyword_pattern = re.compile(rf'{PromptKeyword.AND_ALIGN.value}_(\d+)_(\d+)')
alignment_mask_keyword_pattern = re.compile(rf'{PromptKeyword.AND_MASK_ALIGN.value}_(\d+)_(\d+)')
conciliation_strategies = [e.value for e in ConciliationStrategy]


@dataclasses.dataclass(kw_only=True)
class PromptExpr(abc.ABC):
    weight: float
    conciliation: Optional[ConciliationStrategy]
    conciliation_args: Optional[Tuple[int, ...]] = None
    local_transform: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    @abc.abstractmethod
    def accept(self, visitor, *args, **kwargs) -> Any:
        pass


@dataclasses.dataclass
class LeafPrompt(PromptExpr):
    prompt: str

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_leaf_prompt(self, *args, **kwargs)


@dataclasses.dataclass
class CompositePrompt(PromptExpr):
    children: List[PromptExpr]

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_composite_prompt(self, *args, **kwargs)


class FlatSizeVisitor:
    def visit_leaf_prompt(self, that: LeafPrompt) -> int:
        return 1

    def visit_composite_prompt(self, that: CompositePrompt) -> int:
        return sum(child.accept(self) for child in that.children) if that.children else 0


def parse_root(string: str) -> CompositePrompt:
    tokens = tokenize(string)
    prompts = parse_prompts(tokens)
    return CompositePrompt(
        weight=1.0,
        conciliation=None,
        conciliation_args=None,
        local_transform=None,
        children=prompts,
    )


def parse_prompts(tokens: List[str], *, nested: bool = False) -> List[PromptExpr]:
    prompts = [parse_prompt(tokens, first=True, nested=nested)]
    while tokens:
        if nested and tokens[0] in [']']:
            break

        prompts.append(parse_prompt(tokens, first=False, nested=nested))

    return prompts


def parse_prompt(tokens: List[str], *, first: bool, nested: bool = False) -> PromptExpr:
    conciliation_args = None
    if not first:
        keyword, conciliation_args = parse_prompt_keyword(tokens[0])
        if keyword is not None:
            tokens.pop(0)
            prompt_type = keyword
        else:
            prompt_type = PromptKeyword.AND
    else:
        keyword = PromptKeyword.AND
        prompt_type = keyword

    affine_transform = parse_affine_transform(tokens)

    tokens_copy = tokens.copy()
    if tokens_copy and tokens_copy[0] == '[':
        tokens_copy.pop(0)
        prompts = parse_prompts(tokens_copy, nested=True)
        if tokens_copy:
            assert tokens_copy.pop(0) == ']'
        if len(prompts) > 1:
            tokens[:] = tokens_copy
            weight = parse_weight(tokens)
            conciliation = conciliation_from_keyword(prompt_type)
            return CompositePrompt(
                weight=weight,
                conciliation=conciliation,
                conciliation_args=conciliation_args,
                local_transform=affine_transform,
                children=prompts,
            )

    prompt_text, weight = parse_prompt_text(tokens, nested=nested)
    conciliation = conciliation_from_keyword(prompt_type)
    return LeafPrompt(
        weight=weight,
        conciliation=conciliation,
        conciliation_args=conciliation_args,
        local_transform=affine_transform,
        prompt=prompt_text,
    )


def parse_prompt_keyword(token: str) -> Tuple[Optional[PromptKeyword], Optional[Tuple[int, ...]]]:
    if token in prompt_keywords:
        return PromptKeyword(token), None

    alignment_match = alignment_keyword_pattern.fullmatch(token)
    if alignment_match:
        detail, structure = (int(alignment_match.group(1)), int(alignment_match.group(2)))
        return PromptKeyword.AND_ALIGN, (detail, structure)

    mask_match = alignment_mask_keyword_pattern.fullmatch(token)
    if mask_match:
        detail, structure = (int(mask_match.group(1)), int(mask_match.group(2)))
        return PromptKeyword.AND_MASK_ALIGN, (int(mask_match.group(1)), int(mask_match.group(2)))

    return None, None


def conciliation_from_keyword(keyword: PromptKeyword) -> Optional[ConciliationStrategy]:
    if keyword == PromptKeyword.AND:
        return None
    if keyword == PromptKeyword.AND_PERP:
        return ConciliationStrategy.PERPENDICULAR
    if keyword == PromptKeyword.AND_SALT:
        return ConciliationStrategy.SALIENCE_MASK
    if keyword == PromptKeyword.AND_TOPK:
        return ConciliationStrategy.SEMANTIC_GUIDANCE
    if keyword == PromptKeyword.AND_ALIGN:
        return ConciliationStrategy.ALIGNMENT_BLEND
    if keyword == PromptKeyword.AND_MASK_ALIGN:
        return ConciliationStrategy.ALIGNMENT_MASK
    return None


def parse_prompt_text(tokens: List[str], *, nested: bool = False) -> Tuple[str, float]:
    parts: List[str] = []
    depth_square = 0
    depth_brace = 0
    depth_paren = 0

    while tokens:
        token = tokens[0]
        keyword, _ = parse_prompt_keyword(token)

        if token == '[':
            depth_square += 1
            parts.append(tokens.pop(0))
            continue
        if token == ']':
            if depth_square == 0:
                if nested:
                    break
            else:
                depth_square = max(0, depth_square - 1)
                parts.append(tokens.pop(0))
            continue

        if (
            token == ','
            and nested
            and depth_square == 0
            and depth_brace == 0
            and depth_paren == 0
        ):
            tokens.pop(0)
            break

        if (
            keyword is not None
            and depth_square == 0
            and depth_brace == 0
            and depth_paren == 0
        ):
            break

        part = tokens.pop(0)
        parts.append(part)
        if any(ch in part for ch in '{}()[]'):
            depth_square, depth_brace, depth_paren = _update_delimiter_depths(
                part, depth_square, depth_brace, depth_paren
            )

    text = ''.join(parts)
    prompt_text, weight = extract_prompt_and_weight(text)
    return prompt_text, weight


def _update_delimiter_depths(
    fragment: str,
    depth_square: int,
    depth_brace: int,
    depth_paren: int,
) -> Tuple[int, int, int]:
    i = 0
    while i < len(fragment):
        ch = fragment[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '[':
            depth_square += 1
        elif ch == ']':
            depth_square = max(0, depth_square - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        i += 1

    return depth_square, depth_brace, depth_paren


def parse_weight(tokens: List[str]) -> float:
    weight = 1.
    if len(tokens) >= 2 and tokens[0] == ':' and is_float(tokens[1]):
        tokens.pop(0)
        weight = float(tokens.pop(0))
    return weight


def tokenize(s: str):
    prompt_keywords_regex = '|'.join(rf'\b{keyword}\b' for keyword in prompt_keywords)
    alignment_regex = rf'{PromptKeyword.AND_ALIGN.value}_\d+_\d+|{PromptKeyword.AND_MASK_ALIGN.value}_\d+_\d+'
    transform_regex = '|'.join(rf'\b{keyword}\b' for keyword in affine_transforms_keys())
    pattern = rf'(\[|\]|:|,|{prompt_keywords_regex}|{alignment_regex}|{transform_regex})'
    return [s for s in re.split(pattern, s) if s.strip()]


def affine_transforms_keys() -> Iterable[str]:
    return affine_transforms_mapping().keys()


def affine_transforms_mapping():
    return {
        'ROTATE': lambda angle=0, *_: rotation_matrix(angle),
        'SLIDE': lambda x=0, y=0, *_: translation_matrix(x, y),
        'SCALE': lambda x=1, y=None, *_: scale_matrix(x, x if y is None else y),
        'SHEAR': lambda x=0, y=None, *_: shear_matrix(x, y if y is not None else x),
    }


def rotation_matrix(angle: float):
    radians = angle * 2 * math.pi
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return [
        [cos_v, -sin_v, 0.0],
        [sin_v, cos_v, 0.0],
        [0.0, 0.0, 1.0],
    ]


def translation_matrix(x: float, y: float):
    return [
        [1.0, 0.0, x],
        [0.0, 1.0, y],
        [0.0, 0.0, 1.0],
    ]


def scale_matrix(x: float, y: float):
    return [
        [x, 0.0, 0.0],
        [0.0, y, 0.0],
        [0.0, 0.0, 1.0],
    ]


def shear_matrix(x: float, y: float):
    return [
        [1.0, math.tan(x * 2 * math.pi), 0.0],
        [math.tan(y * 2 * math.pi), 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def parse_affine_transform(tokens: List[str]):
    tokens_copy = tokens.copy()
    if tokens_copy and not tokens_copy[0].strip():
        tokens_copy.pop(0)

    matrices = []
    mapping = affine_transforms_mapping()

    while tokens_copy and tokens_copy[0] in mapping:
        transform_key = tokens_copy.pop(0)
        if not tokens_copy or tokens_copy.pop(0) != '[':
            break

        args = []
        if tokens_copy and tokens_copy[0] != ']':
            arg_token = tokens_copy.pop(0)
            if arg_token.strip():
                try:
                    args = [float(component.strip()) for component in arg_token.split(',')]
                except ValueError:
                    break
            else:
                args = []

        if not tokens_copy or tokens_copy.pop(0) != ']':
            break

        matrices.append(mapping[transform_key](*args))

        if tokens_copy and not tokens_copy[0].strip():
            tokens_copy.pop(0)
        tokens[:] = tokens_copy

    if not matrices:
        return None

    transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    for matrix in reversed(matrices):
        transform = multiply_affine(transform, matrix)

    return tuple(tuple(value for value in row) for row in transform)


def multiply_affine(a: List[List[float]], b: List[List[float]]):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(len(a))
    ]


def make_alignment_keyword(detail: int, structure: int) -> str:
    return f'{PromptKeyword.AND_ALIGN.value}_{detail}_{structure}'


def make_alignment_mask_keyword(detail: int, structure: int) -> str:
    return f'{PromptKeyword.AND_MASK_ALIGN.value}_{detail}_{structure}'


def is_float(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


def extract_prompt_and_weight(text: str) -> Tuple[str, float]:
    text = text or ''

    if ':' not in text and '：' not in text:
        return text, 1.0

    colon_index = _find_last_top_level_colon(text)
    if colon_index is not None:
        trailing = text[colon_index + 1 :].strip()
        if trailing and webui_prompt_parser.RE_NUMERIC_FULL.fullmatch(trailing):
            try:
                return text[:colon_index], float(trailing)
            except ValueError:
                pass

    parts = webui_prompt_parser._split_top_level_colon_keep_empty(text)
    if len(parts) >= 2:
        weight_candidate = parts[-1].strip()
        if weight_candidate and webui_prompt_parser.RE_NUMERIC_FULL.fullmatch(weight_candidate):
            colon_index = _find_last_top_level_colon(text)
            if colon_index is None:
                colon_index = text.rfind(':')
            if colon_index is not None:
                prompt = text[:colon_index]
                try:
                    return prompt, float(weight_candidate)
                except ValueError:
                    pass

    fallback_colon = text.rfind(':')
    if fallback_colon >= 0:
        trailing = text[fallback_colon + 1 :].strip()
        if trailing and webui_prompt_parser.RE_NUMERIC_FULL.fullmatch(trailing):
            try:
                return text[:fallback_colon], float(trailing)
            except ValueError:
                pass

    weight_from_lark = _detect_weight_with_lark(text)
    if weight_from_lark is not None:
        colon_index = _find_last_top_level_colon(text)
        if colon_index is None:
            colon_index = text.rfind(':')
            if colon_index < 0:
                return text, weight_from_lark
        if colon_index is not None:
            prompt = text[:colon_index]
            return prompt, weight_from_lark

    return text, 1.0


def _detect_weight_with_lark(text: str) -> Optional[float]:
    normalized = webui_prompt_parser._apply_and(text)
    try:
        tree = webui_prompt_parser.schedule_parser.parse(normalized)
    except lark.LarkError:
        return None

    prompt_node = _first_prompt_node(tree)
    if prompt_node is None:
        return None

    relevant_children = [child for child in prompt_node.children if not _is_whitespace(child)]
    if len(relevant_children) != 1:
        return None

    weighted_node = relevant_children[0]
    if not isinstance(weighted_node, lark.Tree) or weighted_node.data != 'weighted':
        return None

    number_token = weighted_node.children[-1]
    try:
        return float(str(number_token))
    except ValueError:
        return None


def _find_last_top_level_colon(text: str) -> Optional[int]:
    depth_square = depth_brace = depth_paren = 0
    i = 0
    positions: List[int] = []

    while i < len(text):
        ch = text[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '[':
            depth_square += 1
        elif ch == ']':
            depth_square = max(0, depth_square - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == ':' and depth_square == depth_brace == depth_paren == 0:
            positions.append(i)

        i += 1

    if not positions:
        return None

    return positions[-1]


def _first_prompt_node(tree: lark.Tree) -> Optional[lark.Tree]:
    for child in tree.children:
        if isinstance(child, lark.Tree) and child.data == 'prompt':
            return child
    return None


def _is_whitespace(node: Any) -> bool:
    return isinstance(node, lark.Token) and node.type == 'WHITESPACE'


if __name__ == '__main__':
    res = parse_root('''
    hello
    AND_PERP [
        arst
        AND defg : 2
        AND_SALT [
            very nested huh? what do you say :.0
        ]
    ]
    ''')
    pass
