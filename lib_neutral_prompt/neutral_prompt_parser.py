import abc
import dataclasses
import math
import re
from enum import Enum
from typing import List, Tuple, Any, Optional, Iterable


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
    text = ''
    depth = 0
    weight = 1.
    while tokens:
        next_keyword, _ = parse_prompt_keyword(tokens[0])
        if tokens[0] == ']':
            if depth == 0:
                if nested:
                    break
            else:
                depth -= 1
        elif tokens[0] == '[':
            depth += 1
        elif tokens[0] == ':':
            if len(tokens) >= 2 and is_float(tokens[1].strip()):
                if len(tokens) < 3 or parse_prompt_keyword(tokens[2])[0] is not None or tokens[2] == ']' and depth == 0:
                    tokens.pop(0)
                    weight = float(tokens.pop(0).strip())
                    break
        elif next_keyword is not None:
            break

        text += tokens.pop(0)

    return text, weight


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
    pattern = rf'(\[|\]|:|{prompt_keywords_regex}|{alignment_regex}|{transform_regex})'
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
