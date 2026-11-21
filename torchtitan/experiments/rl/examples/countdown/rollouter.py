# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.countdown.data import CountdownDataset
from torchtitan.experiments.rl.examples.countdown.env import CountdownEnv
from torchtitan.experiments.rl.examples.countdown.rubric import RewardCountdown
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rubrics import Rubric


class CountdownRollouter(Rollouter):
    """Single-turn Countdown with a binary reward; validation on the held-out head."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: CountdownDataset.Config = field(
            default_factory=CountdownDataset.Config
        )
        validation_dataset: CountdownDataset.Config = field(
            default_factory=lambda: CountdownDataset.Config(validation=True)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardCountdown.Config(weight=1.0)],
                error_reward=0.0,
            )
        )
        message_env: CountdownEnv.Config = field(default_factory=CountdownEnv.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_rollout_tokens=512, max_num_turns=1
            )
        )
        advantage: AdvantageEstimator.Config = field(
            default_factory=lambda: AdvantageEstimator.Config(
                should_std_normalize=False
            )
        )
