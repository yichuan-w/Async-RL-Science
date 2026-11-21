# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.countdown.data import CountdownSample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn

_ALLOWED = re.compile(r"^[\d\s+\-*/().]+$")


def extract_answer(text: str) -> str | None:
    found = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return found[-1].strip() if found else None


def score_countdown(text: str, numbers: tuple[int, ...], target: int) -> float:
    """1.0 iff the last <answer> is a + - * / expression using each number exactly
    once and evaluating to ``target`` (the reference ``compute_reward``)."""
    expr = extract_answer(text)
    if expr is None or not _ALLOWED.match(expr) or "**" in expr or "//" in expr:
        return 0.0
    if Counter(int(x) for x in re.findall(r"\d+", expr)) != Counter(numbers):
        return 0.0
    try:
        return 1.0 if abs(eval(expr, {"__builtins__": {}}, {}) - target) < 1e-9 else 0.0
    except Exception:
        return 0.0


class RewardCountdown(RewardFn):
    """Binary Countdown reward on the full completion text."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: CountdownSample) -> float:
        if not rollout.turns:
            return 0.0
        message = rollout.turns[-1].completion_message or {}
        # The renderer may split a <think> block into a reasoning field; the answer
        # can only be in what follows it, but scoring the whole text is robust to
        # either parse.
        text = "".join(
            str(message.get(key) or "")
            for key in ("reasoning_content", "reasoning", "content")
        )
        return score_countdown(text, env_input.numbers, env_input.target)
