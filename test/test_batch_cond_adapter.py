import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from lib_neutral_prompt.cfg_denoiser_hijack import BatchCondAdapter, ReindexedCondInfo
from modules import prompt_parser


def _make_multicond(weights):
    schedules = [prompt_parser.ScheduledPromptConditioning(5, torch.zeros(1, 4))]
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
