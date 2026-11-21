# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixed-age batches: the buffer's random sample order, staggered generator
syncs, and ISGridLoss's per-token age weight."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from torchtitan.experiments.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.losses.is_grid import ISGridLoss
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)


def _taken_order(sample_order: str, n: int = 12) -> list[int]:
    async def run() -> list[int]:
        buffer = RolloutGroupWorkBuffer.Config(sample_order=sample_order).build(
            max_active_rollout_groups=n
        )
        for gid in range(n):
            await buffer.add_work(RolloutGroupWork(group_id=gid, sample=object()))
            await buffer.claim_next()
            await buffer.finalize_work(RolloutGroup(group_id=gid, rollouts=[]))
        return [(await buffer.take_finalized()).group_id for _ in range(n)]

    return asyncio.run(run())


def test_fifo_order_unchanged():
    assert _taken_order("fifo") == list(range(12))


def test_random_order_takes_each_group_once():
    order = _taken_order("random")
    assert sorted(order) == list(range(12))
    assert order != list(range(12))


def test_lifo_takes_newest_first():
    assert _taken_order("lifo") == list(range(11, -1, -1))


def test_sample_order_validated():
    with pytest.raises(ValueError):
        RolloutGroupWorkBuffer.Config(sample_order="newest")


def test_router_pulls_only_the_given_generators():
    gens = [MagicMock() for _ in range(3)]
    for g in gens:
        g.pull_model_state_dict.call = AsyncMock()
    router = InterGeneratorRouter.Config().build(generators=gens)
    asyncio.run(router.pull_model_state_dict(policy_version=7, generator_indices=[1]))
    assert [g.pull_model_state_dict.call.await_count for g in gens] == [0, 1, 0]
    asyncio.run(router.pull_model_state_dict(policy_version=8))
    assert [g.pull_model_state_dict.call.await_count for g in gens] == [1, 2, 1]


@pytest.mark.parametrize("stride", [0, 11])
def test_generators_due_for_sync(stride):
    fake = SimpleNamespace(
        config=SimpleNamespace(
            async_loop=SimpleNamespace(
                weight_sync_interval=65, weight_sync_phase_stride=stride
            )
        ),
        generator_router=SimpleNamespace(num_generators=6),
    )
    due = {v: Controller._generators_due_for_sync(fake, v) for v in range(1, 131)}
    for i in range(6):
        syncs = [v for v, gens in due.items() if i in gens]
        # Every generator is refreshed every 65 steps; with a stride, at its own phase.
        assert [b - a for a, b in zip(syncs, syncs[1:])] == [65]
        assert syncs[0] == 65 - (i * stride) % 65 if stride else syncs[0] == 65


# ---- ISGridLoss age weight ----

V = 6
TRAINER_LOGITS = torch.tensor([0.5, -0.2, 1.1, 0.0, -0.7, 0.3], dtype=torch.float64)
SAMPLER_LOGP = torch.log_softmax(
    TRAINER_LOGITS + torch.tensor([0.4, -0.3, -0.5, 0.2, 0.6, -0.1], dtype=torch.float64),
    -1,
)


def _expected_logit_grad(config: dict, age: float) -> torch.Tensor:
    """E_{y~q}[d loss / d logits] at one state of the given policy age, with the
    sampler's full vocab as its top-k (no tail)."""
    loss_fn = ISGridLoss.Config(**config).build()
    head_logp, head_ids = torch.topk(SAMPLER_LOGP, V)
    total = torch.zeros(V, dtype=torch.float64)
    for y in range(V):
        logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
        loss, _ = loss_fn(
            logits,
            torch.tensor([[y]]),
            1.0,
            generator_logprobs=SAMPLER_LOGP[y].view(1, 1),
            advantages=torch.ones(1, 1, dtype=torch.float64),
            loss_mask=torch.tensor([[True]]),
            topk_token_ids=head_ids.view(1, 1, -1),
            topk_logprobs=head_logp.view(1, 1, -1),
            positions=torch.zeros(1, 1, dtype=torch.long),
            policy_age=torch.tensor([[age]]),
        )
        loss.backward()
        total += SAMPLER_LOGP[y].exp() * logits.grad.view(-1)
    return total


@pytest.mark.parametrize("age_weight", ["linear", "inv"])
def test_state_only_age_weight_keeps_centering(age_weight):
    """A constant advantage carries no learning signal; SC with an age weight
    still leaves no drift toward the sampler, while plain PG does."""
    sc = dict(score_centering=True, age_weight=age_weight, age_weight_a0=64.0)
    assert _expected_logit_grad(sc, age=20.0).abs().max() < 1e-6
    pg = dict(level="none", age_weight=age_weight, age_weight_a0=64.0)
    assert _expected_logit_grad(pg, age=20.0).abs().max() > 1e-3


def _batch(gen: torch.Generator, S: int = 10, k: int = 4):
    logits = torch.randn(1, S, 20, generator=gen, dtype=torch.float64)
    samp = torch.log_softmax(
        logits + 0.5 * torch.randn(1, S, 20, generator=gen, dtype=torch.float64), -1
    )
    labels = torch.randint(0, 20, (1, S), generator=gen)
    head_logp, head_ids = torch.topk(samp, k)
    kwargs = dict(
        generator_logprobs=samp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        advantages=torch.randn(1, S, generator=gen, dtype=torch.float64),
        loss_mask=torch.ones(1, S, dtype=torch.bool),
        topk_token_ids=head_ids,
        topk_logprobs=head_logp,
        positions=torch.arange(S).view(1, -1),
        policy_age=torch.arange(S).view(1, -1) * 8,
    )
    return logits, labels, kwargs


def _loss_and_grad(config: dict, logits, labels, kwargs):
    x = logits.clone().requires_grad_(True)
    loss, metrics = ISGridLoss.Config(**config).build()(x, labels, 10.0, **kwargs)
    loss.backward()
    return loss.detach(), x.grad, metrics


def test_age_weight_none_is_bitwise_unchanged():
    logits, labels, kwargs = _batch(torch.Generator().manual_seed(0))
    base = dict(score_centering=True, kl_mask_threshold=0.3)
    a = _loss_and_grad(base, logits, labels, kwargs)
    b = _loss_and_grad(dict(base, age_weight="none"), logits, labels, kwargs)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_age_weight_renormalizes_to_mean_one():
    """Equal ages: the renormalized weight is 1 everywhere, so the loss is unchanged."""
    logits, labels, kwargs = _batch(torch.Generator().manual_seed(1))
    kwargs = dict(kwargs, policy_age=torch.full_like(kwargs["policy_age"], 30))
    base = dict(score_centering=True)
    a = _loss_and_grad(base, logits, labels, kwargs)
    b = _loss_and_grad(dict(base, age_weight="linear"), logits, labels, kwargs)
    torch.testing.assert_close(a[1], b[1])
    # Mixed ages: old tokens get less gradient, the total weight stays the token count.
    logits, labels, kwargs = _batch(torch.Generator().manual_seed(1))
    _, grad, metrics = _loss_and_grad(
        dict(base, age_weight="linear"), logits, labels, kwargs
    )
    assert grad[0, -1].abs().sum() == 0  # age 72 >= a0: linear weight 0
    assert grad[0, 0].abs().sum() > 0
    assert metrics["age_weight/norm_scale/max"] > 1


def test_countdown_variants_split_capacity_from_age_cap():
    from torchtitan.experiments.rl.examples.countdown import config_registry as cd

    nx = cd._countdown_config(
        method="sc_kl03", async_buffer_steps=32, sample_order="random", stale_drop=False
    ).async_loop
    assert nx.max_active_rollout_groups == 32 * cd.NUM_PROMPTS
    assert nx.max_offpolicy_steps == cd.NEVER_STALE
    assert nx.group_buffer.sample_order == "random"
    a128 = cd.rl_countdown_qwen3_0_6b_async128np_tis_sc().async_loop
    assert a128.max_active_rollout_groups == 128 * cd.NUM_PROMPTS
    assert a128.max_offpolicy_steps == 128 and a128.group_buffer.sample_order == "fifo"


def test_completed_untrained_counts_finalized_and_taken_groups():
    async def run():
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=4)
        for gid in range(3):
            await buffer.add_work(RolloutGroupWork(group_id=gid, sample=object()))
        await buffer.claim_next()
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        assert buffer.num_completed_untrained() == 1
        await buffer.take_finalized()  # handed to the batcher: still untrained
        assert buffer.num_completed_untrained() == 1
        waiter = asyncio.create_task(buffer.wait_for_completed_untrained(2))
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.claim_next()
        await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
        await asyncio.wait_for(waiter, 1)
        await buffer.release_active_groups(1, reason="trained")
        assert buffer.num_completed_untrained() == 1

    asyncio.run(run())


def test_prefill_must_fit_in_the_buffer():
    from torchtitan.experiments.rl.controller import AsyncLoopConfig

    AsyncLoopConfig(
        num_groups_per_train_step=4,
        max_offpolicy_steps=8,
        max_active_rollout_groups=32,
        min_buffered_steps_before_training=8,
    )
    with pytest.raises(ValueError):
        AsyncLoopConfig(
            num_groups_per_train_step=4,
            max_offpolicy_steps=8,
            max_active_rollout_groups=32,
            min_buffered_steps_before_training=9,
        )


def test_async128_variants_prefill():
    from torchtitan.experiments.rl.examples.countdown import config_registry as cd

    for cfg in (
        cd.rl_countdown_qwen3_0_6b_async128s1_sc_kl03(),
        cd._countdown_config(
            method="sc_kl03", async_buffer_steps=128, sample_order="random", prefill_steps=127
        ),
    ):
        a = cfg.async_loop
        assert a.min_buffered_steps_before_training == 127
        assert 127 * a.num_groups_per_train_step <= a.max_active_rollout_groups
    assert cd.rl_countdown_qwen3_0_6b_async128randnp_sc().async_loop.min_buffered_steps_before_training == 0


def test_claims_stay_fifo_and_scale_with_many_waiting_loops():
    """One rollout loop per slot: admitting groups wakes one claimer each, in admission order."""
    import time as _time

    async def run(n: int) -> tuple[list[int], float]:
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=n)
        claimed: list[int] = []

        async def loop():
            work = await buffer.claim_next()
            if work is not None:
                claimed.append(work.group_id)

        tasks = [asyncio.create_task(loop()) for _ in range(n)]
        await asyncio.sleep(0)
        start = _time.perf_counter()
        for gid in range(n):
            await buffer.add_work(RolloutGroupWork(group_id=gid, sample=object()))
        await asyncio.wait_for(asyncio.gather(*tasks), 30)
        return claimed, _time.perf_counter() - start

    claimed, _ = asyncio.run(run(8))
    assert claimed == list(range(8))
    claimed, elapsed = asyncio.run(run(2000))
    assert sorted(claimed) == list(range(2000))
    assert elapsed < 5.0


def test_close_wakes_waiting_claimers():
    async def run():
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=2)
        waiter = asyncio.create_task(buffer.claim_next())
        await asyncio.sleep(0)
        await buffer.close()
        assert await asyncio.wait_for(waiter, 1) is None

    asyncio.run(run())


def test_countdown_seed_reaches_data_sampling_and_buffer():
    from torchtitan.experiments.rl.examples.countdown import config_registry as cd

    base = cd.rl_countdown_qwen3_0_6b_async128np_sc_kl03()
    assert base.generator.debug.seed is None
    assert base.rollouter.train_dataset.shuffle is False
    assert base.async_loop.group_buffer.sample_seed == 0
    for s in (1, 2, 3):
        cfg = getattr(cd, f"rl_countdown_qwen3_0_6b_async128s{s}_sc_kl03")()
        assert cfg.generator.debug.seed == s
        assert cfg.rollouter.train_dataset.shuffle and cfg.rollouter.train_dataset.seed == s
        assert cfg.async_loop.group_buffer.sample_seed == s
        assert cfg.async_loop.num_training_steps == 2400
        assert cfg.async_loop.min_buffered_steps_before_training == 127
        assert cfg.dump_folder.endswith(f"_s{s}_sc_kl03")
