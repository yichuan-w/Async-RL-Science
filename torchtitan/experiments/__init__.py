# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_experiments = frozenset(
    [
        "rl",
        # RL examples own a per-example config_registry under rl/examples/<name>;
        # listed here so `--module <name>` resolves (see ConfigManager).
        "countdown",
    ]
)
