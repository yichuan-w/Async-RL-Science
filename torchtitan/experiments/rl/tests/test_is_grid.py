# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ISGridLoss: the reference repo's drift test, plus the grid's named methods."""

from __future__ import annotations

import math

import pytest
import torch

from torchtitan.experiments.rl.losses.is_grid import ISGridLoss
from torchtitan.experiments.rl.losses.score_centering import ScoreCenteringLoss
from torchtitan.experiments.rl.losses.is_grid import _SFC_KL, _TABLE_1, NAMED_METHODS as _NAMED

# Loss variants the tests exercise beyond the named methods of the post.
_M = dict(_NAMED)
_M["sc_klreg_kl03pg"] = dict(_M["sc_klreg_kl03"], kl_mask_gates_kl_loss=False)
_M["sc_klq10"] = dict(_M["sc"], kl_mask_quantile=0.1)
_M["sc_klreg_klq10"] = dict(_M["sc_klreg"], kl_mask_quantile=0.1)
_M["sc_klconst"] = dict(_M["sc_klreg"], kl_age_schedule="const")
_M["topr_sc_klreg"] = dict(_TABLE_1["topr"], **_SFC_KL)
_M["gspo_sc_klreg"] = dict(_TABLE_1["gspo"], **_SFC_KL)

TRAINER_LOGITS = torch.tensor([0.5, -0.2, 1.1], dtype=torch.float64)
SAMPLER_LOGP = torch.log_softmax(
    TRAINER_LOGITS + torch.tensor([0.1, 0.3, -0.2], dtype=torch.float64), -1
)
CORRECTIONS = {
    "pg": dict(level="none"),
    "tis": dict(level="token", high=2.0, outside="clamp"),
    # This band drops some of the tokens.
    "mis": dict(level="token", low=0.8, high=1.1),
    # The TV band drops the entries whose |q_v - p_v| > 0.05 when moving away from q.
    "dppo": dict(level="token", stat="tv", high=0.05, sign="ppo"),
}


def _token_grad(config: dict, token: int, center: bool) -> torch.Tensor:
    loss_fn = ISGridLoss.Config(score_centering=center, **config).build()
    logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
    # k = V - 1: the tail is one token, so modeling it as rho * p is exact.
    head_logp, head_ids = torch.topk(SAMPLER_LOGP, 2)
    loss, _ = loss_fn(
        logits,
        torch.tensor([[token]]),
        1.0,
        generator_logprobs=SAMPLER_LOGP[token].view(1, 1),
        advantages=torch.ones(1, 1, dtype=torch.float64),
        loss_mask=torch.tensor([[True]]),
        topk_token_ids=head_ids.view(1, 1, -1),
        topk_logprobs=head_logp.view(1, 1, -1),
        positions=torch.zeros(1, 1, dtype=torch.long),
    )
    loss.backward()
    return logits.grad.view(-1)


def _expected_gradient(config: dict, center: bool) -> torch.Tensor:
    grads = torch.stack(
        [_token_grad(config, t, center) for t in range(TRAINER_LOGITS.numel())]
    )
    return (SAMPLER_LOGP.exp()[:, None] * grads).sum(0)


@pytest.mark.parametrize("name", CORRECTIONS)
def test_score_centering_cancels_drift(name):
    assert _expected_gradient(CORRECTIONS[name], True).abs().max() < 1e-6


def test_drift_without_centering():
    assert _expected_gradient(CORRECTIONS["pg"], False).abs().max() > 1e-3


@pytest.mark.parametrize(
    "weight,config",
    [
        ("none", dict(level="none")),
        ("tis", dict(level="token", high=2.0, outside="clamp")),
        ("mis", dict(level="token", low=0.5, high=5.0)),
    ],
)
def test_matches_score_centering_loss(weight, config):
    """Same gradient as the repo's ScoreCenteringLoss on a vocab with a real tail."""
    gen = torch.Generator().manual_seed(0)
    V, S, k = 40, 6, 8
    logits = torch.randn(1, S, V, generator=gen, dtype=torch.float64) * 2
    samp = torch.log_softmax(
        logits + 0.5 * torch.randn(1, S, V, generator=gen, dtype=torch.float64), -1
    )
    labels = torch.randint(0, V, (1, S), generator=gen)
    head_logp, head_ids = torch.topk(samp, k)
    kwargs = dict(
        generator_logprobs=samp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        advantages=torch.randn(1, S, generator=gen, dtype=torch.float64),
        loss_mask=torch.ones(1, S, dtype=torch.bool),
        topk_token_ids=head_ids,
        topk_logprobs=head_logp,
        positions=torch.arange(S).view(1, S),
    )
    grads = []
    for fn in (
        ISGridLoss.Config(score_centering=True, **config).build(),
        ScoreCenteringLoss.Config(weight=weight).build(),
    ):
        x = logits.clone().requires_grad_(True)
        loss, _ = fn(x, labels, float(S), **kwargs)
        loss.backward()
        grads.append(x.grad)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-6, atol=1e-9)


def _seq_loss(config: dict, log_ratios: list[float], advantage: float) -> float:
    """Loss of one 3-token sequence whose trainer logprobs are all log(1/V)."""
    V = 4
    logits = torch.zeros(1, 3, V, dtype=torch.float64)
    logp = -math.log(V)
    loss, _ = ISGridLoss.Config(**config).build()(
        logits,
        torch.zeros(1, 3, dtype=torch.long),
        3.0,
        generator_logprobs=torch.tensor([[logp - r for r in log_ratios]]),
        advantages=torch.full((1, 3), advantage, dtype=torch.float64),
        loss_mask=torch.ones(1, 3, dtype=torch.bool),
        positions=torch.arange(3).view(1, 3),
    )
    return loss.item() / (-advantage * logp)  # the applied weight


def test_sequence_level_weights():
    gspo = dict(level="seq_mean", low=0.9997, high=1.0004, sign="ppo")
    topr = dict(level="seq_sum", high=1.0, outside="clamp", sign="neg")
    # GSPO: mean log ratio inside the band keeps the sequence, outside drops it.
    assert _seq_loss(gspo, [1e-4, -1e-4, 3e-4], 1.0) == pytest.approx(
        math.exp(1e-4), rel=1e-6
    )
    assert _seq_loss(gspo, [0.1, 0.0, 0.0], 1.0) == 0.0
    # TOPR: positive advantages untouched, negative get min(trajectory ratio, 1).
    assert _seq_loss(topr, [0.5, 0.5, 0.5], 1.0) == pytest.approx(1.0)
    assert _seq_loss(topr, [-0.2, -0.2, -0.2], -1.0) == pytest.approx(math.exp(-0.6))
    assert _seq_loss(topr, [0.2, 0.2, 0.2], -1.0) == pytest.approx(1.0)


def test_ppo_band():
    ppo = dict(level="token", low=0.8, high=1.2, sign="ppo")
    # A > 0: ratio above high drops, below low is kept; A < 0 mirrors with dual clip.
    assert _seq_loss(ppo, [0.3, 0.3, 0.3], 1.0) == 0.0
    assert _seq_loss(ppo, [-0.3, -0.3, -0.3], 1.0) == pytest.approx(math.exp(-0.3))
    assert _seq_loss(ppo, [-0.3, -0.3, -0.3], -1.0) == 0.0
    assert _seq_loss(ppo, [0.3, 0.3, 0.3], -1.0) == pytest.approx(math.exp(0.3))
    assert _seq_loss(ppo, [1.2, 1.2, 1.2], -1.0) == 0.0  # dual clip at 3


@pytest.mark.parametrize("threshold", [math.inf, 1e-4])
def test_head_kl_gate_keeps_drift_cancellation(threshold):
    """The head-KL gate depends only on the prefix, not on the sampled token, so
    gated score centering still has zero expected update under a constant reward
    (at 1e-4 the gate drops this prefix, leaving exactly zero)."""
    config = dict(level="none", kl_mask_threshold=threshold)
    assert _expected_gradient(config, True).abs().max() < 1e-6


def test_kl_gate_drops_token_from_gradient():
    gen = torch.Generator().manual_seed(1)
    V, S, k = 30, 4, 6
    logits = torch.randn(1, S, V, generator=gen, dtype=torch.float64)
    # Position 2 gets a sampler far from the trainer; the rest match exactly.
    samp_logits = logits.clone()
    samp_logits[0, 2] += 3 * torch.randn(V, generator=gen, dtype=torch.float64)
    samp = torch.log_softmax(samp_logits, -1)
    labels = torch.randint(0, V, (1, S), generator=gen)
    head_logp, head_ids = torch.topk(samp, k)
    kwargs = dict(
        generator_logprobs=samp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        advantages=torch.ones(1, S, dtype=torch.float64),
        loss_mask=torch.ones(1, S, dtype=torch.bool),
        topk_token_ids=head_ids,
        topk_logprobs=head_logp,
        positions=torch.arange(S).view(1, S),
    )
    grads, metrics = [], []
    for thr in (math.inf, 0.1):
        x = logits.clone().requires_grad_(True)
        loss, m = ISGridLoss.Config(score_centering=True, kl_mask_threshold=thr).build()(
            x, labels, float(S), **kwargs
        )
        loss.backward()
        grads.append(x.grad)
        metrics.append(m)
    assert grads[1][0, 2].abs().max() == 0  # gated position: no gradient
    assert grads[0][0, 2].abs().max() > 0
    torch.testing.assert_close(grads[1][0, [0, 1, 3]], grads[0][0, [0, 1, 3]])
    assert metrics[1]["is_grid/kl_masked_frac"].item() == pytest.approx(1 / S)


def _gated_klreg_grads(advantage: float, extra: dict | None = None):
    """Logit gradients of sc_klreg, sc_klreg_kl03 and sc_klreg_kl03pg on four tokens
    of age 1 where only position 2 has a sampler far enough to be gated at 0.3."""
    NAMED_METHODS = _M

    gen = torch.Generator().manual_seed(1)
    V, S, k = 30, 4, 6
    logits = torch.randn(1, S, V, generator=gen, dtype=torch.float64)
    samp_logits = logits.clone()
    samp_logits[0, 2] += 3 * torch.randn(V, generator=gen, dtype=torch.float64)
    samp = torch.log_softmax(samp_logits, -1)
    labels = torch.randint(0, V, (1, S), generator=gen)
    head_logp, head_ids = torch.topk(samp, k)
    kwargs = dict(
        generator_logprobs=samp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        advantages=torch.full((1, S), advantage, dtype=torch.float64),
        loss_mask=torch.ones(1, S, dtype=torch.bool),
        topk_token_ids=head_ids,
        topk_logprobs=head_logp,
        positions=torch.arange(S).view(1, S),
        policy_age=torch.ones(1, S, dtype=torch.long),
    )
    out = {}
    configs = {n: NAMED_METHODS[n] for n in ("sc_klreg", "sc_klreg_kl03", "sc_klreg_kl03pg")}
    configs.update(extra or {})
    for name, cfg in configs.items():
        x = logits.clone().requires_grad_(True)
        loss, m = ISGridLoss.Config(**cfg).build()(x, labels, float(S), **kwargs)
        loss.backward()
        out[name] = (loss.detach(), x.grad, m)
    return out


def test_kl_gate_on_klreg_default_drops_kl_pull_too():
    out = _gated_klreg_grads(1.0)
    _, g, m = out["sc_klreg_kl03"]
    assert m["is_grid/kl_masked_frac"].item() == pytest.approx(1 / 4)
    assert g[0, 2].abs().max() == 0  # gated: neither score nor KL pull
    # Default kl_mask_gates_kl_loss=True is the loss as it was: bitwise the same
    # as spelling the gate into the KL term by hand.
    NAMED_METHODS = _M

    cfg = dict(NAMED_METHODS["sc_klreg_kl03"])
    assert "kl_mask_gates_kl_loss" not in cfg
    assert ISGridLoss.Config(**cfg).kl_mask_gates_kl_loss is True
    torch.testing.assert_close(g[0, [0, 1, 3]], out["sc_klreg"][1][0, [0, 1, 3]], rtol=0, atol=0)


def test_named_klreg_gate_thresholds():
    NAMED_METHODS = _M

    for tag, t in {"03": 0.3, "01": 0.1, "003": 0.03}.items():
        cfg = NAMED_METHODS[f"sc_klreg_kl{tag}"]
        assert cfg == dict(NAMED_METHODS["sc_klreg"], kl_mask_threshold=t)
        assert ISGridLoss.Config(**cfg).kl_mask_gates_kl_loss is True


def test_kl_gate_pg_only_keeps_kl_pull_on_gated_token():
    out = _gated_klreg_grads(1.0)
    _, g_pg, _ = out["sc_klreg_kl03pg"]
    _, g_all, _ = out["sc_klreg_kl03"]
    assert g_pg[0, 2].abs().max() > 0  # the KL pull survives the gate
    torch.testing.assert_close(g_pg[0, [0, 1, 3]], g_all[0, [0, 1, 3]], rtol=0, atol=0)
    # On the gated token what remains is exactly the KL pull: the same gradient as
    # sc_klreg with zero advantage, beta * (p - q) up to the tail approximation.
    kl_only = _gated_klreg_grads(0.0)["sc_klreg"][1]
    torch.testing.assert_close(g_pg[0, 2], kl_only[0, 2])


def _quantile_gate_case(config: dict, labels: torch.Tensor):
    """Four positions, full-vocab top-k, sampler drift growing with position."""
    gen = torch.Generator().manual_seed(3)
    V, S = 6, 4
    logits = torch.randn(1, S, V, generator=gen, dtype=torch.float64)
    noise = torch.randn(1, S, V, generator=gen, dtype=torch.float64)
    samp = torch.log_softmax(logits + noise * torch.arange(1, S + 1).view(1, S, 1) * 0.4, -1)
    head_logp, head_ids = torch.topk(samp, V)
    x = logits.clone().requires_grad_(True)
    loss, m = ISGridLoss.Config(**config).build()(
        x,
        labels,
        float(S),
        generator_logprobs=samp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
        advantages=torch.ones(1, S, dtype=torch.float64),
        loss_mask=torch.ones(1, S, dtype=torch.bool),
        topk_token_ids=head_ids,
        topk_logprobs=head_logp,
        positions=torch.arange(S).view(1, S),
        policy_age=torch.ones(1, S, dtype=torch.long),
    )
    loss.backward()
    return loss.detach(), x.grad, m, samp


def test_kl_quantile_gate_drops_top_fraction_and_keeps_centering():
    cfg = dict(level="none", score_centering=True, kl_mask_quantile=0.25)
    labels = torch.zeros(1, 4, dtype=torch.long)
    _, g, m, samp = _quantile_gate_case(cfg, labels)
    assert m["is_grid/kl_masked_frac"].item() == pytest.approx(0.25)
    assert (g[0].abs().amax(-1) == 0).sum() == 1  # exactly one position gated
    # The gate depends only on the prefixes, so with a constant reward the
    # expected update stays zero: average each position's gradient over its
    # sampled token with the others held fixed.
    for s in range(4):
        expected = torch.zeros(6, dtype=torch.float64)
        for y in range(6):
            lab = labels.clone()
            lab[0, s] = y
            expected += samp[0, s, y].exp() * _quantile_gate_case(cfg, lab)[1][0, s]
        assert expected.abs().max() < 1e-6


def test_kl_quantile_zero_is_bitwise_off():
    labels = torch.tensor([[1, 2, 3, 4]])
    base = _quantile_gate_case(dict(level="none", score_centering=True), labels)
    off = _quantile_gate_case(
        dict(level="none", score_centering=True, kl_mask_quantile=0.0), labels
    )
    torch.testing.assert_close(off[0], base[0], rtol=0, atol=0)
    torch.testing.assert_close(off[1], base[1], rtol=0, atol=0)
    with pytest.raises(ValueError):
        ISGridLoss.Config(score_centering=True, kl_mask_quantile=0.1, kl_mask_threshold=0.3).build()


def test_kl_quantile_with_penalty_respects_gate_switch():
    NAMED_METHODS = _M

    labels = torch.zeros(1, 4, dtype=torch.long)
    _, g, m, _ = _quantile_gate_case(NAMED_METHODS["sc_klreg_klq10"], labels)
    assert m["is_grid/kl_masked_frac"].item() == pytest.approx(0.25)
    assert m["is_grid/kl_mask_threshold"].item() > 0
    assert (g[0].abs().amax(-1) == 0).sum() == 1  # score and KL pull both gone
    keep = dict(NAMED_METHODS["sc_klreg_klq10"], kl_mask_gates_kl_loss=False)
    assert (_quantile_gate_case(keep, labels)[1][0].abs().amax(-1) == 0).sum() == 0
    assert NAMED_METHODS["sc_klq10"] == dict(NAMED_METHODS["sc"], kl_mask_quantile=0.1)


def test_named_kl_gate_methods():
    NAMED_METHODS = _M

    assert NAMED_METHODS["sc_kl01"] == dict(
        level="none", score_centering=True, kl_mask_threshold=0.1
    )
    with pytest.raises(ValueError):
        ISGridLoss.Config(kl_mask_threshold=0.1).build()  # head KL needs top-k


@pytest.mark.parametrize("age,beta", [(0, 0.0), (1, 0.3), (4, 0.15)])
def test_sfc_kl_term_is_beta_times_kl_gradient(age, beta):
    """With zero advantage only the KL pull remains; its logit gradient is exactly
    beta_t * (p - q) (the gradient of beta_t * KL(q || p)) when the top-k leaves a
    single tail token, and beta_t = 0.3 / sqrt(age), 0 at age 0."""
    head_logp, head_ids = torch.topk(SAMPLER_LOGP, 2)
    logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
    loss, metrics = ISGridLoss.Config(
        score_centering=True, kl_coef=0.3, kl_age_schedule="inv_sqrt"
    ).build()(
        logits,
        torch.tensor([[0]]),
        1.0,
        generator_logprobs=SAMPLER_LOGP[0].view(1, 1),
        advantages=torch.zeros(1, 1, dtype=torch.float64),
        loss_mask=torch.tensor([[True]]),
        topk_token_ids=head_ids.view(1, 1, -1),
        topk_logprobs=head_logp.view(1, 1, -1),
        positions=torch.zeros(1, 1, dtype=torch.long),
        policy_age=torch.tensor([[age]]),
    )
    loss.backward()
    p = torch.softmax(TRAINER_LOGITS, -1)
    q = SAMPLER_LOGP.exp()
    torch.testing.assert_close(logits.grad.view(-1), beta * (p - q))
    assert metrics["sfc_kl/beta_mean"].item() == pytest.approx(beta)


@pytest.mark.parametrize("age", [0, 1, 9])
def test_sfc_kl_const_applies_beta_at_every_age(age):
    NAMED_METHODS = _M

    head_logp, head_ids = torch.topk(SAMPLER_LOGP, 2)
    logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
    loss, metrics = ISGridLoss.Config(**NAMED_METHODS["sc_klconst"]).build()(
        logits,
        torch.tensor([[0]]),
        1.0,
        generator_logprobs=SAMPLER_LOGP[0].view(1, 1),
        advantages=torch.zeros(1, 1, dtype=torch.float64),
        loss_mask=torch.tensor([[True]]),
        topk_token_ids=head_ids.view(1, 1, -1),
        topk_logprobs=head_logp.view(1, 1, -1),
        positions=torch.zeros(1, 1, dtype=torch.long),
        policy_age=torch.tensor([[age]]),
    )
    loss.backward()
    p = torch.softmax(TRAINER_LOGITS, -1)
    q = SAMPLER_LOGP.exp()
    torch.testing.assert_close(logits.grad.view(-1), 0.3 * (p - q))
    assert metrics["sfc_kl/beta_mean"].item() == pytest.approx(0.3)


def test_sfc_kl_composes_with_sequence_weights():
    NAMED_METHODS = _M

    for name in ("sc_klreg", "topr_sc_klreg", "gspo_sc_klreg"):
        ISGridLoss.Config(**_M[name]).build()
    with pytest.raises(ValueError):
        ISGridLoss.Config(kl_coef=0.3).build()  # needs score centering


def test_trajectory_ratio_metrics():
    """Two packed sequences: the trajectory-ratio sums are prod_t p/q per sequence."""
    gen = torch.Generator().manual_seed(1)
    V, S = 5, 5
    logits = torch.randn(1, S, V, generator=gen, dtype=torch.float64)
    labels = torch.randint(0, V, (1, S), generator=gen)
    logp = torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1))[..., 0]
    logq = logp + torch.tensor([[0.1, -0.2, 0.3, 0.05, -0.4]], dtype=torch.float64)
    _, m = ISGridLoss.Config(level="none").build()(
        logits,
        labels,
        float(S),
        generator_logprobs=logq,
        advantages=torch.ones(1, S, dtype=torch.float64),
        loss_mask=torch.ones(1, S, dtype=torch.bool),
        positions=torch.tensor([[0, 1, 2, 0, 1]]),
    )
    seq_lr = torch.tensor([-(0.1 - 0.2 + 0.3), -(0.05 - 0.4)])
    assert m["is_grid/num_seqs_sum"].item() == 2
    assert m["is_grid/traj_w_sum"].item() == pytest.approx(seq_lr.exp().sum().item(), rel=1e-5)
    assert m["is_grid/traj_w_sq_sum"].item() == pytest.approx(
        (2 * seq_lr).exp().sum().item(), rel=1e-5
    )
    assert m["is_grid/traj_log_ratio_abs/max"].item() == pytest.approx(0.35, rel=1e-5)


@pytest.mark.parametrize("age,beta", [(0, 0.0), (1, 0.3 / 4), (16, 0.3 / 4), (64, 0.3 / 8)])
def test_sfc_kl_age_floor_caps_beta(age, beta):
    head_logp, head_ids = torch.topk(SAMPLER_LOGP, 2)
    logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
    loss, metrics = ISGridLoss.Config(
        score_centering=True, kl_coef=0.3, kl_age_schedule="inv_sqrt", kl_age_floor=16.0
    ).build()(
        logits,
        torch.tensor([[0]]),
        1.0,
        generator_logprobs=SAMPLER_LOGP[0].view(1, 1),
        advantages=torch.zeros(1, 1, dtype=torch.float64),
        loss_mask=torch.tensor([[True]]),
        topk_token_ids=head_ids.view(1, 1, -1),
        topk_logprobs=head_logp.view(1, 1, -1),
        positions=torch.zeros(1, 1, dtype=torch.long),
        policy_age=torch.tensor([[age]]),
    )
    loss.backward()
    p = torch.softmax(TRAINER_LOGITS, -1)
    q = SAMPLER_LOGP.exp()
    torch.testing.assert_close(logits.grad.view(-1), beta * (p - q))
    assert metrics["sfc_kl/beta_mean"].item() == pytest.approx(beta)


@pytest.mark.parametrize("in_head", [True, False])
@pytest.mark.parametrize("adv,kl_coef", [(0.7, 0.0), (-0.4, 0.0), (0.0, 0.3)])
def test_force_metrics_match_autograd_logit_grad(in_head, adv, kl_coef):
    """force/pg_logit_l1_mean and force/kl_logit_l1_mean equal the L1 norm of the
    loss gradient w.r.t. the logits when only that term is active (label inside and
    outside the sampler's top-k)."""
    head_logp, head_ids = torch.topk(SAMPLER_LOGP, 2)
    label = int(head_ids[0]) if in_head else int(torch.topk(SAMPLER_LOGP, 3).indices[-1])
    logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
    loss, metrics = ISGridLoss.Config(score_centering=True, kl_coef=kl_coef).build()(
        logits,
        torch.tensor([[label]]),
        1.0,
        generator_logprobs=SAMPLER_LOGP[label].view(1, 1),
        advantages=torch.full((1, 1), adv, dtype=torch.float64),
        loss_mask=torch.tensor([[True]]),
        topk_token_ids=head_ids.view(1, 1, -1),
        topk_logprobs=head_logp.view(1, 1, -1),
        positions=torch.zeros(1, 1, dtype=torch.long),
        policy_age=torch.tensor([[1]]),
    )
    loss.backward()
    l1 = logits.grad.abs().sum().item()
    if kl_coef > 0:
        assert metrics["force/kl_logit_l1_mean"].item() == pytest.approx(l1, rel=1e-6)
        assert metrics["force/pg_logit_l1_mean"].item() == 0.0
    else:
        assert metrics["force/pg_logit_l1_mean"].item() == pytest.approx(l1, rel=1e-6)
        assert "force/kl_logit_l1_mean" not in metrics


def test_kl_loss_on_gated_only_swaps_terms():
    """kl03sw: the gated token gets only the KL pull, the kept ones only the policy term."""
    NAMED_METHODS = _M

    sw = dict(NAMED_METHODS["sc_klreg_kl03pg"], kl_loss_on_gated_only=True)
    pg_only = dict(NAMED_METHODS["sc_klreg_kl03"], kl_coef=0.0)
    g_sw = _gated_klreg_grads(1.0, extra={"sw": sw, "pg_only": pg_only})
    kl_only = _gated_klreg_grads(0.0)["sc_klreg"][1]
    torch.testing.assert_close(g_sw["sw"][1][0, 2], kl_only[0, 2])
    torch.testing.assert_close(g_sw["sw"][1][0, [0, 1, 3]], g_sw["pg_only"][1][0, [0, 1, 3]])
    with pytest.raises(ValueError):
        ISGridLoss.Config(score_centering=True, kl_coef=0.3, kl_loss_on_gated_only=True).build()


def test_force_metric_keys_without_topk():
    """A microbatch without sampler top-k emits the same force keys (zeros), so the
    all-reduce over ranks sees one key set."""
    logits = TRAINER_LOGITS.clone().view(1, 1, -1).requires_grad_(True)
    _, metrics = ISGridLoss.Config(score_centering=True, kl_coef=0.3).build()(
        logits,
        torch.tensor([[0]]),
        1.0,
        generator_logprobs=SAMPLER_LOGP[0].view(1, 1),
        advantages=torch.ones(1, 1, dtype=torch.float64),
        loss_mask=torch.tensor([[True]]),
        positions=torch.zeros(1, 1, dtype=torch.long),
        policy_age=torch.tensor([[1]]),
    )
    assert metrics["force/pg_logit_l1_mean"].item() == 0.0
    assert metrics["force/kl_logit_l1_mean"].item() == 0.0
