# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.countdown.data import (
    CountdownSample,
    instruction,
    SYSTEM_PROMPT,
)


class CountdownEnv(MessageEnv):
    """Single-turn: system + user prompt, one assistant answer."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: CountdownSample) -> None:
        del config
        self._sample = env_input

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": instruction(
                        list(self._sample.numbers), self._sample.target
                    ),
                },
            ]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        del completion_message
        return MessageEnvStepOutput(done=True)
