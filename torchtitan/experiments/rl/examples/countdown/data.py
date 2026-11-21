# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Countdown (Gandhi et al., 2024): reach a target from four numbers with + - * /.

Problems are generated exactly as in the score-centering reference
(github.com/martin-marek/score-centering, ``envs/countdown.py``): a seeded stream of
solvable instances, the first ``num_validation`` held out for validation.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from torchtitan.config import Configurable


class _CyclingDataset(Configurable):
    """Provides an endless, resumable stream over a finite sample list."""

    def __init__(
        self,
        samples: list,
        *,
        seed: int,
        shuffle: bool,
    ) -> None:
        if not samples:
            raise ValueError("dataset must contain at least one sample")
        self._samples = samples
        self._rng = random.Random(seed)
        self._shuffle = shuffle
        self._order = list(range(len(samples)))
        if shuffle:
            self._rng.shuffle(self._order)
        self._position = 0

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self._position == len(self._order):
            # Rollout production consumes an endless stream; crossing the dataset
            # boundary starts a new epoch. Training reshuffles; validation does not.
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._position = 0
        sample_index = self._order[self._position]
        self._position += 1
        return self._samples[sample_index]

    def state_dict(self) -> dict:
        """Snapshot row order and position so resume continues the same stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state returned by `state_dict`."""
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = state_dict["position"]


OPERATIONS = ("+", "-", "*", "/")

SYSTEM_PROMPT = (
    "You are a helpful assistant. You first thinks about the reasoning process in "
    "the mind and then provides the user with the answer."
)


def instruction(numbers: list[int], target: int) -> str:
    return (
        f" Using the numbers {numbers}, create an equation that equals {target}. You "
        "can use basic arithmetic operations (+, -, *, /) and each number can only be "
        "used once. Show your work in <think> </think> tags. And return the final "
        "answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class CountdownSample:
    numbers: tuple[int, ...]
    target: int
    solution: str


def _build_random_expr(rng: random.Random, numbers: list[int]) -> tuple[Fraction, str]:
    """Combine ``numbers`` pairwise in random order and ops -> (value, expr)."""
    items = [(Fraction(n), str(n)) for n in numbers]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        av, ae = items[i]
        bv, be = items[j]
        op = rng.choice(list(OPERATIONS))
        if op == "/" and bv == 0:
            op = "+"
        val = {"+": av + bv, "-": av - bv, "*": av * bv, "/": av / bv if bv else av}[op]
        items = [items[k] for k in range(len(items)) if k not in (i, j)]
        items.append((val, f"({ae} {op} {be})"))
    return items[0]


def generate(
    rng: random.Random,
    num_operands: int = 4,
    min_number: int = 1,
    max_number: int = 100,
    max_target: int = 1000,
    max_tries: int = 200,
) -> CountdownSample:
    """One solvable instance; re-rolls until the value is an integer in range."""
    for _ in range(max_tries):
        numbers = [rng.randint(min_number, max_number) for _ in range(num_operands)]
        val, expr = _build_random_expr(rng, numbers)
        if val.denominator == 1 and 1 <= val <= max_target:
            return CountdownSample(
                numbers=tuple(numbers),
                target=int(val),
                solution=expr[1:-1] if expr[0] == "(" else expr,
            )
    raise RuntimeError("no in-range integer target found in max_tries")


def _problems(config: "CountdownDataset.Config") -> list[CountdownSample]:
    rng = random.Random(config.generation_seed)
    problems = [
        generate(
            rng,
            num_operands=config.num_operands,
            min_number=config.min_number,
            max_number=config.max_number,
            max_target=config.max_target,
        )
        for _ in range(config.dataset_size)
    ]
    # The reference shuffles the dataset with the run seed before holding out eval.
    random.Random(config.split_seed).shuffle(problems)
    return problems


class CountdownDataset(_CyclingDataset):
    """Training split: the generated problems minus the held-out validation head."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dataset_size: int = 20_000
        num_validation: int = 64
        num_operands: int = 4
        min_number: int = 1
        max_number: int = 100
        max_target: int = 1000
        generation_seed: int = 0
        split_seed: int = 0
        seed: int = 42
        shuffle: bool = False
        """The reference walks the shuffled problems in order."""
        validation: bool = False
        """Serve the held-out head instead of the training rows."""

    def __init__(self, config: Config) -> None:
        problems = _problems(config)
        head, rest = problems[: config.num_validation], problems[config.num_validation :]
        super().__init__(
            head if config.validation else rest,
            seed=config.seed,
            shuffle=config.shuffle and not config.validation,
        )
