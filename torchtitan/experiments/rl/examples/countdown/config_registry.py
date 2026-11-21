# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-0.6B on Countdown: the score-centering paper's small-model benchmark.

Marek & Ryabinin (arXiv 2609.20807), Section 5 / Appendix B: REINFORCE with
+1/-1 group-centered rewards, plain SGD at lr 1e-2, 64 prompts x 8 completions
per step, 512 tokens, one optimizer step per batch, and one correction method on
top (``ISGridLoss``).

Configs are ``rl_countdown_qwen3_0_6b_<variant>_<method>``, ``<method>`` a key of
``ISGridLoss`` ``NAMED_METHODS``:

- ``stale``: the paper's staleness experiment. The samplers get new weights only
  every ``STALENESS + 1`` steps (``weight_sync_interval``), so trained rollouts
  are 0..``STALENESS`` versions old whatever the sampler speed.
- ``async<N>`` and the ``_LAG_STUDY`` variants: weights pushed every step, a
  rollout buffer of ``N`` steps (``max_offpolicy_steps``). ``N`` caps the age; the
  actual age depends on how far generation runs ahead of training, which with six
  samplers per trainer keeps the buffer full.

The blog post's runs use one 8-GPU host per run: a 2-GPU trainer and six
single-GPU samplers (``repro/run.sh``).
"""

from __future__ import annotations

import math

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.components.work_buffer import RolloutGroupWorkBuffer
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.countdown.data import CountdownDataset
from torchtitan.experiments.rl.examples.countdown.rollouter import CountdownRollouter
from torchtitan.experiments.rl.losses import ISGridLoss
from torchtitan.experiments.rl.losses.is_grid import NAMED_METHODS
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.models.qwen3 import model_registry

NUM_PROMPTS = 64
GROUP_SIZE = 8
# The reference caps prompt + completion at 512; the chat prompt is ~130 tokens.
MAX_TOTAL_TOKENS = 512
MAX_RESPONSE_TOKENS = 384
STALENESS = 64
ASYNC_BUFFER_STEPS = 32
TOPK = 128
NEVER_STALE = 1_000_000


def _countdown_config(
    *,
    method: str = "tis_sc",
    staleness: int = 0,
    async_buffer_steps: int = 0,
    num_steps: int = 1200,
    num_prompts: int = NUM_PROMPTS,
    buffer_groups: int | None = None,
    lr: float = 1e-2,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    validation_interval: int = 50,
    sample_order: str = "fifo",
    weight_sync_phase_stride: int = 0,
    stale_drop: bool = True,
    prefill_steps: int = 0,
    seed: int = 0,
) -> Controller.Config:
    """``staleness``: sync generator weights every ``staleness + 1`` steps.
    ``async_buffer_steps``: sync every step, with that many steps of run-ahead.
    ``num_prompts`` / ``buffer_groups``: batch size and rollout-buffer size in groups
    (default ``async_buffer_steps * num_prompts``) for the batch-size study.
    ``lr``: SGD learning rate (the paper's 1e-2 by default).
    ``max_response_tokens``: generation budget (384 = the paper's 512-token context).
    ``sample_order``: the rollout buffer's take order (``random`` mixes policy ages
    within a batch). ``weight_sync_phase_stride``: stagger the generators' syncs
    (with ``staleness``), which also mixes ages within a batch.
    ``stale_drop``: False never drops a group for its age; the buffer keeps its
    capacity (``async_buffer_steps * num_prompts`` groups), so generation still
    waits when it is full. ``prefill_steps``: steps of completed groups the buffer
    holds before training starts (``min_buffered_steps_before_training``).
    ``seed``: 0 keeps the defaults (problems walked in order, unseeded sampling).
    A positive seed shuffles the training problems with it, seeds vLLM sampling (per
    group and sample) and the buffer's random take; the model starts from the same
    checkpoint either way. Capacity (``max_active_rollout_groups``) and the age cap
    (``max_offpolicy_steps``) are separate settings already; this lifts only the cap."""
    loss = ISGridLoss.Config(**NAMED_METHODS[method])
    if staleness:
        tag = f"s{staleness}"
        # One step of run-ahead; +2 covers it and requests decoding across a sync.
        max_offpolicy_steps, active_groups = staleness + 2, 2 * NUM_PROMPTS
    else:
        tag = f"async{async_buffer_steps}"
        active_groups = buffer_groups or max(async_buffer_steps, 1) * num_prompts
        max_offpolicy_steps = max(async_buffer_steps, active_groups // num_prompts)
        if num_prompts != NUM_PROMPTS:
            tag = f"async{max_offpolicy_steps}b{num_prompts}"
        if not stale_drop:
            # An age no group reaches in a run: nothing is stale-dropped.
            max_offpolicy_steps = NEVER_STALE
            tag += "_nx"
    if lr != 1e-2:
        tag += f"_lr{lr:g}"
    if max_response_tokens != MAX_RESPONSE_TOKENS:
        tag += f"_t{max_response_tokens}"
    if sample_order != "fifo":
        tag += f"_{sample_order}"
    if weight_sync_phase_stride:
        tag += f"_phase{weight_sync_phase_stride}"
    if seed:
        tag += f"_s{seed}"
    return Controller.Config(
        model_spec=model_registry(
            "0.6B",
            attn_backend="varlen",
            converters=[LMHeadCastConverter.Config()],
        ),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        dump_folder=f"outputs/rl/countdown_0_6b_{tag}_{method}",
        async_loop=AsyncLoopConfig(
            num_training_steps=num_steps,
            num_groups_per_train_step=num_prompts,
            group_size=GROUP_SIZE,
            max_offpolicy_steps=max_offpolicy_steps,
            max_active_rollout_groups=active_groups,
            weight_sync_interval=staleness + 1,
            weight_sync_phase_stride=weight_sync_phase_stride,
            min_buffered_steps_before_training=prefill_steps,
            group_buffer=RolloutGroupWorkBuffer.Config(
                sample_order=sample_order, sample_seed=seed
            ),
            # The paper trains on every group, zero-variance ones included
            # (they add tokens to the loss denominator, not gradient).
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False
            ),
            validation=ValidationConfig(
                num_samples=64, interval=validation_interval, group_size=8, temperature=1.0
            ),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=8192),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=CountdownRollouter.Config(
            advantage=AdvantageEstimator.Config(scale=2.0),
            train_dataset=(
                CountdownDataset.Config(shuffle=True, seed=seed)
                if seed
                else CountdownDataset.Config()
            ),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(
            enable_wandb=True, enable_tensorboard=True, wandb_project="score-centering"
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(
                param_groups=[
                    ParamGroupConfig(
                        pattern=r".*",
                        optimizer_name="SGD",
                        # Fused SGD has no DTensor sharding strategy.
                        optimizer_kwargs={"lr": lr, "fused": False, "foreach": True},
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0, min_lr_factor=1.0
            ),
            # optax.sgd in the reference: no clipping.
            training=TrainingConfig(max_norm=math.inf),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1, tensor_parallel_degree=1
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=100_000,
                last_save_model_only=True,
            ),
            # Sequence-level corrections (GSPO, TOPR) need whole sequences per
            # call, so no ChunkedLossWrapper; 0.6B logits fit.
            loss=loss,
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1, tensor_parallel_degree=1
            ),
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            debug=DebugConfig(seed=seed or None),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
                num_topk_logprobs=TOPK if loss.score_centering else 0,
            ),
        ),
    )


def rl_countdown_qwen3_0_6b() -> Controller.Config:
    """On-policy Countdown with TIS + score centering (the paper's recommendation)."""
    return _countdown_config()


def _registered(method: str, *, staleness: int = 0, async_buffer_steps: int = 0):
    def config_fn() -> Controller.Config:
        return _countdown_config(
            method=method, staleness=staleness, async_buffer_steps=async_buffer_steps
        )

    if staleness:
        name, doc = "stale", f"sampler refreshed every {staleness + 1} steps"
    else:
        name = f"async{async_buffer_steps}"
        doc = f"async, {async_buffer_steps}-step rollout buffer"
    config_fn.__name__ = f"rl_countdown_qwen3_0_6b_{name}_{method}"
    config_fn.__doc__ = f"Countdown, {doc}, correction {method}."
    return config_fn


# The async settings of the blog post "Beyond Score Centering: What Else Can We
# Optimize in Async RL?". Weights are pushed to the samplers after every step;
# ``async_buffer_steps`` is the age cap K of the rollout buffer.
_LAG_STUDY = {
    # 128-step buffer, oldest-first take, no prefill: training starts on an empty
    # buffer and the batch age climbs to 127 as generation outruns training.
    "async128np": dict(async_buffer_steps=128),
    # The same buffer with groups taken uniformly at random (mean age ~52).
    "async128randnp": dict(async_buffer_steps=128, sample_order="random"),
    # The same buffer with the newest finished group taken first. Stale groups are
    # dropped only when taken, so older groups stay in the buffer.
    "async128lifonp": dict(async_buffer_steps=128, sample_order="lifo"),
    # Random take at 4x the learning rate, seeds 0-2.
    "async128randnplr4": dict(async_buffer_steps=128, sample_order="random", lr=4e-2),
    **{
        f"async128randnplr4s{s}": dict(
            async_buffer_steps=128, sample_order="random", lr=4e-2, seed=s
        )
        for s in (1, 2)
    },
    # Oldest-first take from a buffer prefilled with 127 steps of groups, so every
    # batch after the first 127 steps is at age 127; 2,400 steps, seeds 1-3.
    **{
        f"async128s{s}": dict(async_buffer_steps=128, prefill_steps=127, num_steps=2400, seed=s)
        for s in (1, 2, 3)
    },
}
for _method in NAMED_METHODS:
    for _variant, _kwargs in _LAG_STUDY.items():

        def _fn(_m=_method, _k=_kwargs):
            return _countdown_config(method=_m, **_k)

        _fn.__name__ = f"rl_countdown_qwen3_0_6b_{_variant}_{_method}"
        _fn.__doc__ = f"Countdown lag study {_variant}, {_method}."
        globals()[_fn.__name__] = _fn

for _method in NAMED_METHODS:
    for _fn in (
        _registered(_method, staleness=STALENESS),
        _registered(_method, async_buffer_steps=ASYNC_BUFFER_STEPS),
    ):
        globals()[_fn.__name__] = _fn
