"""Regression tests for fixed half-group MIRA reflection triggers."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from mira.training import triggers
from verl import DataProto


TRIGGER_TEXT = "Double-check the visual evidence."


@pytest.fixture(autouse=True)
def configure_trigger_pool(monkeypatch):
    monkeypatch.setattr(
        triggers, "triggers", {"english": [TRIGGER_TEXT], "chinese": [TRIGGER_TEXT]}
    )


def object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def make_batch(rollout_n, structured_content=True):
    prompts = []
    for sample_type in [1, 0, 1]:
        prompt = [{"role": "user", "content": "Answer the image question."}]
        if sample_type == 1:
            content = "<think>A misleading image description."
            if structured_content:
                content = [{"type": "text", "text": content}]
            prompt.append({"role": "assistant", "content": content})
        prompts.append(prompt)
    original = deepcopy(prompts)
    batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(3, 1)},
        non_tensors={
            "prompt": object_array(prompts),
            "raw_prompt": object_array(deepcopy(prompts)),
            "extra_info": object_array([{"type": kind} for kind in [1, 0, 1]]),
        },
    ).repeat(repeat_times=rollout_n, interleave=True)
    return batch, original


def apply_triggers(batch, rollout_n, step, enable_schedule, mode="part"):
    return triggers.add_correction_trigger(
        batch,
        rollout_n=rollout_n,
        global_step=step,
        total_step=100,
        warmup_ratio=0.3,
        start_ratio=0.6,
        final_ratio=0.4,
        mode=mode,
        enable_schedule=enable_schedule,
    )


def assert_group_counts(batch, original, rollout_n, expected_augmented_count):
    for group, sample_type in enumerate([1, 0, 1]):
        start = group * rollout_n
        rows = batch.non_tensor_batch["prompt"][start : start + rollout_n]
        count = sum(TRIGGER_TEXT in str(prompt) for prompt in rows)
        assert count == (expected_augmented_count if sample_type == 1 else 0)
        for index, prompt in enumerate(rows, start):
            assert batch.non_tensor_batch["raw_prompt"][index] == prompt
            if sample_type == 0:
                assert prompt == original[group]
        assert TRIGGER_TEXT not in str(original[group])


@pytest.mark.parametrize("step", [0, 30, 60, 100])
@pytest.mark.parametrize("rollout_n", [4, 8])
@pytest.mark.parametrize("structured_content", [False, True])
def test_disabled_schedule_keeps_exactly_half_at_every_step(
    step, rollout_n, structured_content
):
    batch, original = make_batch(rollout_n, structured_content)
    result = apply_triggers(batch, rollout_n, step, enable_schedule=False)
    assert_group_counts(result, original, rollout_n, rollout_n // 2)


@pytest.mark.parametrize("step,expected_count", [(0, 4), (60, 3), (100, 3)])
def test_explicit_schedule_preserves_historical_counts(step, expected_count):
    batch, original = make_batch(8)
    result = apply_triggers(batch, 8, step, enable_schedule=True)
    assert_group_counts(result, original, 8, expected_count)


def test_fixed_half_group_rejects_odd_rollout_count():
    batch, _ = make_batch(3)
    with pytest.raises(ValueError, match="requires an even rollout_n"):
        apply_triggers(batch, 3, 60, enable_schedule=False)


def test_all_mode_remains_active_when_schedule_is_disabled(monkeypatch):
    monkeypatch.setattr(triggers.random, "random", lambda: 0.0)
    batch, original = make_batch(8)
    result = apply_triggers(batch, 8, 100, enable_schedule=False, mode="all")
    assert_group_counts(result, original, 8, 8)
