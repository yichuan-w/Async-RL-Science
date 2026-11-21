# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""REINFORCE with an importance-sampling correction from one grid, plus optional
score centering: the comparison of Marek & Ryabinin, "Score Centering Stabilizes
Off-policy Reinforcement Learning" (arXiv 2609.20807), Table 1.

Ported from the reference ``losses.py`` (github.com/martin-marek/score-centering).
Every correction is REINFORCE with a detached weight ``w(r)`` on the trainer to
sampler ratio ``r = p/q``, per token or per sequence. The named methods are points
on one grid (unlisted fields keep their defaults ``level=token stat=ratio low=0
high=inf outside=drop sign=both``)::

    PG          level=none
    naive IS    level=token
    TIS         high=2.0 outside=clamp
    MIS         low=0.5 high=5.0
    PPO         low=0.8 high=1.2 sign=ppo
    DAPO        low=0.8 high=1.28 sign=ppo
    GSPO        level=seq_mean low=0.9997 high=1.0004 sign=ppo
    TOPR        level=seq_sum high=1.0 outside=clamp sign=neg
    DPPO        stat=tv high=0.2 sign=ppo

``score_centering`` subtracts the expected score under the effective sampling mass
``q * w`` (top-k head from the sampler, tail modeled as ``rho * p``), which cancels
the drift a train/inference mismatch adds to the update. It composes with every
token-level correction (a ``stat=tv`` band tests each vocab entry's ``|q_v - p_v|``)
and with PG.

Sequence-level corrections need whole sequences in one call: use this loss without
``ChunkedLossWrapper``. Sequences are delimited by ``positions`` resetting to 0.
The summed per-token loss is divided by ``global_valid_tokens`` (all completion
tokens of the step), the reference's ``mask.sum()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss
from torchtitan.config import CompileConfig
from torchtitan.experiments.rl.losses.dppo import entropy_from_logits
from torchtitan.experiments.rl.losses.score_centering import _gather_logprobs

# PPO's ratio bound for negative advantages (dual clip, arXiv 1912.09729).
DUAL_CLIP = 3.0
_LEVELS = ("none", "token", "seq_mean", "seq_sum")
_STATS = ("ratio", "tv")
_OUTSIDE = ("clamp", "drop")
_SIGNS = ("both", "neg", "pos", "ppo")
_KL_STATS = ("head", "sampled")
_KL_AGE_SCHEDULES = ("inv_sqrt", "const")
_AGE_WEIGHTS = ("none", "linear", "inv")

# Table 1 of the paper (the reference README's grid), plus the score-centered
# compositions it plots: name -> ISGridLoss.Config kwargs.
_TABLE_1 = {
    "pg": dict(level="none"),
    "naive_is": dict(level="token"),
    "tis": dict(level="token", high=2.0, outside="clamp"),
    "mis": dict(level="token", low=0.5, high=5.0),
    "ppo": dict(level="token", low=0.8, high=1.2, sign="ppo"),
    "dapo": dict(level="token", low=0.8, high=1.28, sign="ppo"),
    "gspo": dict(level="seq_mean", low=0.9997, high=1.0004, sign="ppo"),
    "topr": dict(level="seq_sum", high=1.0, outside="clamp", sign="neg"),
    "dppo": dict(level="token", stat="tv", high=0.2, sign="ppo"),
}
NAMED_METHODS = {
    **_TABLE_1,
    "sc": dict(_TABLE_1["pg"], score_centering=True),
    "tis_sc": dict(_TABLE_1["tis"], score_centering=True),
    "mis_sc": dict(_TABLE_1["mis"], score_centering=True),
}
# KL mask: score centering, leaving out every position whose head KL between the
# sampler and the trainer, KL(mu(.|s) || pi(.|s)), exceeds the threshold.
KL_GATE_THRESHOLDS = {"003": 0.03, "01": 0.1, "03": 0.3, "1": 1.0}
NAMED_METHODS.update(
    {
        f"sc_kl{tag}": dict(_TABLE_1["pg"], score_centering=True, kl_mask_threshold=t)
        for tag, t in KL_GATE_THRESHOLDS.items()
    }
)
# KL penalty (after Yihua Zhang's score-function-centering experiments,
# normaluhr.github.io): score centering minus beta_a * KL(mu || pi) per token,
# beta_a = 0.3 / sqrt(a) for policy age a (0 for fresh tokens).
_SFC_KL = dict(score_centering=True, kl_coef=0.3, kl_age_schedule="inv_sqrt")
NAMED_METHODS["sc_klreg"] = dict(_TABLE_1["pg"], **_SFC_KL)
# The recipe: KL penalty and KL mask together; a masked position drops both terms.
NAMED_METHODS.update(
    {
        f"sc_klreg_kl{tag}": dict(_TABLE_1["pg"], **_SFC_KL, kl_mask_threshold=t)
        for tag, t in {"03": 0.3, "01": 0.1, "003": 0.03}.items()
    }
)


class ISGridLoss(BaseLoss):
    """REINFORCE + one IS-grid correction + optional score centering."""

    accepts_topk = True
    accepts_policy_age = True

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        level: str = "none"
        """``none`` (plain PG) | ``token`` | ``seq_mean`` (mean log r, GSPO) |
        ``seq_sum`` (sum log r, the trajectory ratio, TOPR)."""

        stat: str = "ratio"
        """The band tests the ratio ``r`` or the TV distance ``|q - p|`` (DPPO)."""

        low: float = 0.0
        high: float = math.inf
        """Band ``[low, high]``; 0 / inf leave a side open."""

        outside: str = "drop"
        """Outside the band: ``clamp`` to the nearest bound, or ``drop`` (w = 0)."""

        sign: str = "both"
        """Which advantages get corrected: ``both`` | ``neg`` | ``pos`` | ``ppo``
        (block only moves away from q: A > 0 keeps (0, high], A < 0 [low, 3])."""

        score_centering: bool = False
        """Subtract the expected score under ``q * w``; needs sampler top-k
        (``SamplingConfig.num_topk_logprobs``)."""

        kl_mask_threshold: float = math.inf
        """Drop a token from the gradient (score and centering term alike) when
        its sampler-vs-trainer KL exceeds this. It stays in the loss denominator,
        like a dropped token under MIS. inf disables the gate."""

        kl_mask_quantile: float = 0.0
        """Gate by proportion instead: drop the fraction ``q`` of trained tokens
        with the highest per-token KL, i.e. the threshold is the ``1 - q``
        quantile of ``token_kl`` over the valid tokens (in the loss mask and with
        sampler top-k). The quantile is taken within each loss call, so per
        microbatch, not over the whole step's batch. The gate still depends only
        on the prefix, so centering is kept. 0 disables it; exclusive with
        ``kl_mask_threshold``."""

        kl_mask_stat: str = "head"
        """Per-token KL for the gate. ``head``: the full next-token KL(q || p)
        from the sampler's top-k, ``sum_H q log(q/p) + q_tail log(q_tail/p_tail)``
        (needs ``score_centering`` for the top-k). ``sampled``: the k3 estimate
        ``r - 1 - log r`` on the sampled token, ``r = p/q``."""

        kl_coef: float = 0.0
        """beta_0 of an added ``beta_t * grad KL(q || p)`` pull toward the sampler,
        applied to every trained token independent of its advantage (SFC+KL). The
        gradient is ``beta_t * sum_v q_v grad log p_v`` over the sampler's top-k with
        the trainer-shaped tail (the score-centering coefficients without IS
        weights); needs ``score_centering``."""

        kl_loss_on_gated_only: bool = False
        """Apply the ``kl_coef`` pull only to the tokens the KL gate drops, so each
        trained token gets exactly one of the two terms: the policy term where the
        sampler is close, the pull toward it where it is far. Needs a gate and
        ``kl_mask_gates_kl_loss=False``."""

        kl_age_floor: float = 1.0
        """``inv_sqrt`` uses ``max(age, kl_age_floor)`` for ages >= 1, capping beta_t at
        ``kl_coef / sqrt(kl_age_floor)``. Under FIFO with a prefilled buffer the first
        K steps train on version-0 data at ages 1..K-1, so beta_t would reach
        ``kl_coef`` at step 2 while the steady-state age is K-1; a floor of K-1 keeps
        beta_t at its steady-state value there. 1 is the plain schedule."""

        kl_mask_gates_kl_loss: bool = True
        """Whether the KL gate (``kl_mask_threshold`` or ``kl_mask_quantile``) also drops a gated token's ``kl_coef``
        pull. False keeps the pull on every trained token, so the gate removes
        only the score and centering term."""

        kl_age_schedule: str = "inv_sqrt"
        """``inv_sqrt``: ``beta_t = kl_coef / sqrt(age)`` and 0 for an age-0 token;
        ``const``: ``beta_t = kl_coef``. ``age`` is the token's policy age (trainer
        version minus the version that sampled it)."""

        age_weight: str = "none"
        """Coarse down-weighting of older tokens by policy age ``a``: ``none``;
        ``linear``, ``c = max(0, 1 - a / age_weight_a0)``; ``inv``,
        ``c = 1 / (1 + a / age_weight_a0)``. ``c`` multiplies the token's whole loss
        term and depends only on the state (not on the sampled token), so score
        centering stays exact. The weights are renormalized to mean 1 over each
        microbatch's trained tokens: the step's total weight is unchanged (the
        learning rate is not), and only the split between old and fresh tokens
        moves. A single loss call sees one microbatch, so the renormalization is
        per microbatch; it equals the step-wide one when microbatches mix ages
        alike (``age_weight/norm_scale/max`` against ``1 / age_weight/raw_mean``
        shows how far they differ)."""

        age_weight_a0: float = 64.0
        """Age scale of ``age_weight``: where ``linear`` reaches 0, where ``inv``
        halves."""

    def __init__(
        self, config: Config, *, compile_config: CompileConfig | None = None
    ) -> None:
        del compile_config
        for name, allowed in (
            ("level", _LEVELS),
            ("stat", _STATS),
            ("outside", _OUTSIDE),
            ("sign", _SIGNS),
        ):
            if getattr(config, name) not in allowed:
                raise ValueError(
                    f"ISGridLoss.{name} must be one of {allowed}, got {getattr(config, name)!r}"
                )
        if config.kl_mask_stat not in _KL_STATS:
            raise ValueError(
                f"ISGridLoss.kl_mask_stat must be one of {_KL_STATS}, got {config.kl_mask_stat!r}"
            )
        if not 0.0 <= config.kl_mask_quantile < 1.0:
            raise ValueError(
                f"ISGridLoss.kl_mask_quantile must be in [0, 1), got {config.kl_mask_quantile}"
            )
        if config.kl_mask_quantile > 0 and config.kl_mask_threshold < math.inf:
            raise ValueError("set only one of kl_mask_threshold and kl_mask_quantile")
        if (
            (config.kl_mask_threshold < math.inf or config.kl_mask_quantile > 0)
            and config.kl_mask_stat == "head"
            and not config.score_centering
        ):
            raise ValueError("kl_mask_stat='head' needs score_centering (sampler top-k)")
        if config.kl_age_schedule not in _KL_AGE_SCHEDULES:
            raise ValueError(
                f"ISGridLoss.kl_age_schedule must be one of {_KL_AGE_SCHEDULES}, "
                f"got {config.kl_age_schedule!r}"
            )
        if config.age_weight not in _AGE_WEIGHTS:
            raise ValueError(
                f"ISGridLoss.age_weight must be one of {_AGE_WEIGHTS}, got {config.age_weight!r}"
            )
        if config.age_weight_a0 <= 0:
            raise ValueError(
                f"ISGridLoss.age_weight_a0 must be positive, got {config.age_weight_a0}"
            )
        if config.kl_loss_on_gated_only and (
            config.kl_mask_gates_kl_loss
            or (config.kl_mask_threshold == math.inf and config.kl_mask_quantile == 0)
        ):
            raise ValueError(
                "kl_loss_on_gated_only needs a KL gate and kl_mask_gates_kl_loss=False"
            )
        if config.kl_age_floor < 1:
            raise ValueError(f"ISGridLoss.kl_age_floor must be >= 1, got {config.kl_age_floor}")
        if config.kl_coef > 0 and not config.score_centering:
            raise ValueError("kl_coef needs score_centering (sampler top-k)")
        if config.stat == "tv" and config.level == "seq_sum":
            raise ValueError("ISGridLoss: seq_sum requires stat=ratio")
        self.c = config

    def _log_mass(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        positive: torch.Tensor,
        tv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """log(q * w(p/q)): the effective sampling mass under this correction
        (-inf where w = 0). The IS weight is the mass at (p = r, q = 1); the top-k
        tail factor ``rho * w(1/rho)`` is the mass at (p = 1, q = rho)."""
        c = self.c
        log_p, log_q = torch.broadcast_tensors(log_p, log_q)
        neg_inf = torch.full_like(log_p, -math.inf)
        lo = log_q + math.log(c.low) if c.low > 0 else neg_inf
        hi = log_q + math.log(c.high) if c.high < math.inf else -neg_inf
        if c.stat == "tv":
            in_band = tv <= c.high
            if c.sign == "ppo":
                in_band = in_band | torch.where(
                    positive, log_p <= log_q, log_p >= log_q
                )
        elif c.sign == "ppo":
            in_band = torch.where(
                positive,
                log_p <= hi,
                (log_p >= lo) & (log_p <= log_q + math.log(DUAL_CLIP)),
            )
        else:
            in_band = (log_p >= lo) & (log_p <= hi)
        outside = (
            torch.maximum(torch.minimum(log_p, hi), lo)
            if c.outside == "clamp"
            else neg_inf
        )
        log_m = torch.where(in_band, log_p, outside)
        if c.sign == "neg":
            log_m = torch.where(positive, log_q, log_m)
        if c.sign == "pos":
            log_m = torch.where(positive, log_m, log_q)
        return log_m

    def _is_weight(
        self,
        log_ratio: torch.Tensor,
        positive: torch.Tensor,
        tv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_ratio = log_ratio.clamp(-20.0, 20.0)
        return torch.exp(
            self._log_mass(log_ratio, torch.zeros_like(log_ratio), positive, tv)
        )

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        *,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        topk_token_ids: torch.Tensor | None = None,
        topk_logprobs: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        metric_denominator: float | None = None,
        policy_age: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        c = self.c
        if topk_token_ids is None:
            topk_token_ids = labels.new_zeros((*labels.shape, 0))
            topk_logprobs = generator_logprobs.new_zeros((*labels.shape, 0))
        assert topk_logprobs is not None
        ids = torch.cat([labels.unsqueeze(-1), topk_token_ids.long()], dim=-1)
        logp_all = _gather_logprobs(logits, ids)
        logp = logp_all[..., 0]
        logp_head = logp_all[..., 1:]
        logq = generator_logprobs.float()

        mask = loss_mask & torch.isfinite(logq) & torch.isfinite(logp.detach())
        maskf = mask.float()
        logq = torch.where(mask, logq, torch.zeros_like(logq))
        positive = advantages >= 0

        with torch.no_grad():
            logp_d = torch.where(mask, logp.detach(), torch.zeros_like(logq))
            log_ratio = logp_d - logq
            tv = (logq.exp() - logp_d.exp()).abs()
            if c.level == "token":
                w = self._is_weight(log_ratio, positive, tv)
            else:
                w = torch.ones_like(logq)

        score = w * logp
        has_topk = torch.zeros_like(mask)
        coef_l1 = torch.zeros_like(logq)
        head_tv = torch.zeros_like(logq)
        head_mass = torch.zeros_like(logq)
        kl_obj = torch.zeros_like(logq)
        beta = torch.zeros_like(logq)
        with torch.no_grad():
            # Sampled-token k3 estimate of KL(q || p); the head KL replaces it below.
            token_kl = torch.exp(log_ratio) - 1.0 - log_ratio
        if c.score_centering and topk_logprobs.shape[-1] > 0:
            with torch.no_grad():
                has_topk = torch.isfinite(topk_logprobs[..., 0])
                logq_head = torch.where(
                    torch.isfinite(topk_logprobs),
                    topk_logprobs.float(),
                    torch.full_like(topk_logprobs, -math.inf, dtype=torch.float32),
                )
                p_head = logp_head.detach().exp()
                q_head = logq_head.exp()
                head_mass = q_head.sum(-1)
                q_tail = (1.0 - head_mass).clamp(min=0.0)
                rho = q_tail / (1.0 - p_head.sum(-1)).clamp(min=1e-6)
                log_rho = rho.clamp(min=1e-30).log()
                if c.level == "token":
                    # stat=tv tests each vocab entry's own |q_v - p_v| (the tail
                    # as one entry), like the sampled token's.
                    p_tail = (1.0 - p_head.sum(-1)).clamp(min=0.0)
                    alpha = self._log_mass(
                        torch.zeros_like(log_rho),
                        log_rho,
                        positive,
                        (q_tail - p_tail).abs(),
                    ).exp()
                    mass = self._log_mass(
                        logp_head.detach(),
                        logq_head,
                        positive.unsqueeze(-1),
                        (q_head - p_head).abs(),
                    ).exp()
                else:
                    alpha = rho
                    mass = q_head
                residual = mass - alpha.unsqueeze(-1) * p_head
                residual = torch.where(
                    has_topk.unsqueeze(-1), residual, torch.zeros_like(residual)
                )
                coef_l1 = residual.abs().sum(-1)
                head_tv = 0.5 * torch.where(
                    has_topk.unsqueeze(-1), (q_head - p_head).abs(), torch.zeros_like(p_head)
                ).sum(-1)
                if c.kl_mask_stat == "head":
                    head_kl = torch.where(
                        q_head > 0,
                        q_head * (logq_head - logp_head.detach()),
                        torch.zeros_like(q_head),
                    ).sum(-1)
                    p_tail = (1.0 - p_head.sum(-1)).clamp(min=1e-6)
                    tail_kl = torch.where(
                        q_tail > 0,
                        q_tail * (q_tail.clamp(min=1e-30).log() - p_tail.log()),
                        torch.zeros_like(q_tail),
                    )
                    token_kl = torch.where(has_topk, head_kl + tail_kl, torch.zeros_like(head_kl))
            # E_p[log p] term of the reference has zero gradient; omitted.
            score = score - (residual * logp_head).sum(-1)
            if c.kl_coef > 0:
                with torch.no_grad():
                    if c.kl_age_schedule == "const":
                        beta = torch.full_like(logq, c.kl_coef)
                    else:
                        if policy_age is None:
                            raise ValueError("kl_age_schedule='inv_sqrt' needs policy_age")
                        age = policy_age.float()
                        beta = torch.where(
                            age >= 1,
                            c.kl_coef / age.clamp(min=max(c.kl_age_floor, 1.0)).sqrt(),
                            torch.zeros_like(age),
                        )
                    kl_residual = torch.where(
                        has_topk.unsqueeze(-1),
                        q_head - rho.unsqueeze(-1) * p_head,
                        torch.zeros_like(p_head),
                    )
                # Objective beta_t * sum_H sg[q - rho p] log p: its gradient is
                # beta_t * E_q[grad log p] = -beta_t * grad KL(q || p).
                kl_obj = beta * (kl_residual * logp_head).sum(-1)

        kl_kept = torch.ones_like(mask)
        kl_threshold = None
        if c.kl_mask_threshold < math.inf:
            kl_kept = token_kl <= c.kl_mask_threshold
        elif c.kl_mask_quantile > 0:
            # Threshold from this microbatch's valid tokens only.
            with torch.no_grad():
                valid = mask & has_topk if c.kl_mask_stat == "head" else mask
                if valid.any():
                    kl_threshold = torch.quantile(
                        token_kl[valid].float(), 1.0 - c.kl_mask_quantile
                    )
                    kl_kept = token_kl <= kl_threshold
        score = score * maskf * kl_kept.float()
        adv = advantages.float()
        seq_w_metrics = {}
        if c.level in ("seq_mean", "seq_sum") and positions is None:
            raise ValueError("sequence-level corrections need positions")
        if positions is not None:
            with torch.no_grad():
                flat_pos = positions.reshape(-1)
                seg = torch.cumsum((flat_pos == 0).long(), dim=0) - 1
                num_seg = int(seg.max().item()) + 1 if seg.numel() else 0
                seg = seg.clamp(min=0)

                def seg_sum(x: torch.Tensor) -> torch.Tensor:
                    return torch.zeros(
                        num_seg, device=x.device, dtype=torch.float32
                    ).index_add_(0, seg, x.reshape(-1).float())

                tokens = seg_sum(maskf).clamp(min=1.0)
                traj_log_ratio = seg_sum(log_ratio * maskf)
                live = seg_sum(maskf) > 0
                # Trajectory ratio prod_t p/q of every sequence, for any level: its
                # effective sample size (sum w)^2 / (n sum w^2) is computed offline
                # from these sums (they are summed over microbatches and ranks).
                traj_w = torch.where(
                    live, traj_log_ratio.clamp(-20.0, 20.0).exp(), torch.zeros_like(tokens)
                )
                seq_w_metrics = {
                    "is_grid/num_seqs_sum": live.float().sum(),
                    "is_grid/traj_w_sum": traj_w.sum(),
                    "is_grid/traj_w_sq_sum": traj_w.square().sum(),
                    "is_grid/traj_log_ratio_sum": (traj_log_ratio * live).sum(),
                    "is_grid/traj_log_ratio_sq_sum": (traj_log_ratio.square() * live).sum(),
                    "is_grid/traj_log_ratio_abs/max": (traj_log_ratio.abs() * live).max()
                    if num_seg
                    else torch.zeros((), device=maskf.device),
                }
                if c.level in ("seq_mean", "seq_sum"):
                    seq_log_ratio = traj_log_ratio
                    if c.level == "seq_mean":
                        seq_log_ratio = seq_log_ratio / tokens
                    seq_tv = seg_sum(tv * maskf) / tokens
                    # The advantage is constant over a sequence's trained tokens.
                    seq_adv = seg_sum(adv * maskf) / tokens
                    seq_w = self._is_weight(seq_log_ratio, seq_adv >= 0, seq_tv)
                    tok_w = seq_w[seg].reshape(adv.shape)
                    seq_w_metrics["is_grid/seq_masked_frac_sum"] = (
                        ((seq_w == 0) & live).float().sum()
                    )
            if c.level in ("seq_mean", "seq_sum"):
                adv = adv * tok_w

        if c.kl_loss_on_gated_only:
            kl_gate = (~kl_kept).float()
        else:
            kl_gate = kl_kept.float() if c.kl_mask_gates_kl_loss else 1.0
        token_loss = -(adv * score + kl_obj * maskf * kl_gate)
        age_metrics = {}
        if c.age_weight != "none":
            if policy_age is None:
                raise ValueError("age_weight needs policy_age")
            with torch.no_grad():
                age = policy_age.float().clamp(min=0)
                if c.age_weight == "linear":
                    age_w = (1.0 - age / c.age_weight_a0).clamp(min=0.0)
                else:
                    age_w = 1.0 / (1.0 + age / c.age_weight_a0)
                raw = (age_w * maskf).sum()
                norm_scale = maskf.sum() / raw.clamp(min=1e-12)
                age_metrics = {
                    "age_weight/raw_mean": raw / max(metric_denominator or 1, 1)
                    if metric_denominator is not None
                    else raw / max(global_valid_tokens or 1, 1),
                    "age_weight/norm_scale/max": norm_scale,
                }
            token_loss = token_loss * (age_w * norm_scale)
        loss_denominator = (
            max(global_valid_tokens, 1) if global_valid_tokens is not None else 1
        )
        loss = token_loss.sum() / loss_denominator
        metric_denom = (
            max(metric_denominator, 1)
            if metric_denominator is not None
            else loss_denominator
        )

        with torch.no_grad():
            diff = log_ratio * maskf
            trained = mask & has_topk
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": (log_ratio.exp() * maskf).sum() / metric_denom,
                "loss/is_weight_mean": (w * maskf).sum() / metric_denom,
                "is_grid/token_masked_frac": ((w == 0) & mask).float().sum()
                / metric_denom,
                "bit_wise/logprob_diff/abs_mean": diff.abs().sum() / metric_denom,
                "bit_wise/logprob_diff/max": diff.abs().max(),
                # k1 KL(q || p) estimate on sampled tokens, the reference's kl_samp_train.
                "is_grid/kl_samp_train": (-diff).sum() / metric_denom,
                "debug/vllm_local_kl_k3_mean": (torch.exp(diff) - diff - 1.0).sum()
                / metric_denom,
                "is_grid/tv_mean": (tv * maskf).sum() / metric_denom,
            }
            metrics.update(seq_w_metrics)
            metrics.update(age_metrics)
            if c.kl_coef > 0:
                metrics["sfc_kl/beta_mean"] = (beta * maskf).sum() / metric_denom
                metrics["sfc_kl/policy_age_mean"] = (
                    (policy_age.float() * maskf).sum() / metric_denom
                    if policy_age is not None
                    else torch.zeros((), device=maskf.device)
                )
            if c.kl_mask_quantile > 0:
                metrics["is_grid/kl_mask_threshold"] = (
                    kl_threshold
                    if kl_threshold is not None
                    else torch.zeros((), device=maskf.device)
                )
            if c.kl_mask_threshold < math.inf or c.kl_mask_quantile > 0:
                metrics["is_grid/kl_masked_frac"] = (
                    (mask & ~kl_kept).float().sum() / metric_denom
                )
                metrics["is_grid/token_kl_mean"] = (token_kl * maskf).sum() / metric_denom
                metrics["is_grid/token_kl/max"] = (token_kl * maskf).max()
            if c.score_centering:
                # Every rank must emit the same keys (they are all-reduced): zeros on a
                # microbatch without top-k (e.g. a padding microbatch).
                zero = torch.zeros((), device=maskf.device)
                metrics["force/pg_logit_l1_mean"] = zero
                if c.kl_coef > 0:
                    metrics["force/kl_logit_l1_mean"] = zero
            if c.score_centering and topk_logprobs.shape[-1] > 0:
                # L1 norm over the vocabulary of each term's gradient w.r.t. the logits,
                # per trained token (analytic, from d log p_v / dz = e_v - p): the policy
                # term |A| * |d score / dz|_1 on kept tokens, and the KL pull
                # beta * |d (sum_H k_v log p_v) / dz|_1, k = q - rho p. They show which
                # term drives a step.
                p_h = logp_head.detach().exp()
                p_y = logp.detach().exp()
                y_in_head = topk_token_ids.long() == labels.unsqueeze(-1)
                y_in = y_in_head.any(-1)
                c_sum = w - residual.sum(-1)
                head_g = w.unsqueeze(-1) * y_in_head - residual - c_sum.unsqueeze(-1) * p_h
                label_g = torch.where(y_in, torch.zeros_like(p_y), w - c_sum * p_y)
                rest = (1.0 - p_h.sum(-1) - torch.where(y_in, torch.zeros_like(p_y), p_y)).clamp(min=0.0)
                pg_l1 = head_g.abs().sum(-1) + label_g.abs() + c_sum.abs() * rest
                kept_f = maskf * kl_kept.float()
                metrics["force/pg_logit_l1_mean"] = (
                    advantages.float().abs() * pg_l1 * kept_f
                ).sum() / metric_denom
                if c.kl_coef > 0:
                    k_sum = kl_residual.sum(-1)
                    kl_l1 = beta * (
                        (kl_residual - k_sum.unsqueeze(-1) * p_h).abs().sum(-1)
                        + k_sum.abs() * (1.0 - p_h.sum(-1)).clamp(min=0.0)
                    )
                    metrics["force/kl_logit_l1_mean"] = (
                        kl_l1 * maskf * kl_gate
                    ).sum() / metric_denom
            if c.score_centering:
                metrics["sc/topk_missing_frac"] = (
                    (mask & ~has_topk).float().sum() / metric_denom
                )
                metrics["sc/sampler_head_mass_mean"] = (
                    head_mass * trained
                ).sum() / metric_denom
                metrics["sc/centering_coef_l1_mean"] = (
                    coef_l1 * trained
                ).sum() / metric_denom
                # Sampler vs trainer on the sampler's head; the coefficient L1 above
                # is ~1 wherever the head holds all the mass (q_tail = 0, rho = 0)
                # even when p == q, so this is the one to read for mismatch.
                metrics["sc/head_tv_mean"] = (head_tv * trained).sum() / metric_denom
                # Alignment check: the sampled token, when in the sampler's top-k,
                # carries the same logprob there as in generator_logprobs. Every
                # rank must emit the same keys (they are all-reduced), so these
                # are zeros on a microbatch without top-k.
                match = topk_token_ids.long() == labels.unsqueeze(-1)
                in_topk = match.any(-1) & trained
                q_at_label = torch.where(
                    match,
                    topk_logprobs.float(),
                    torch.zeros_like(topk_logprobs, dtype=torch.float32),
                ).sum(-1)
                metrics["sc/label_in_topk_frac"] = in_topk.float().sum() / metric_denom
                metrics["sc/label_topk_logprob_err/max"] = torch.where(
                    in_topk, (q_at_label - logq).abs(), torch.zeros_like(logq)
                ).max()
            entropy = entropy_from_logits(logits)
            if entropy is not None:
                metrics["policy/entropy_avg"] = (entropy * maskf).sum() / metric_denom

        return loss, metrics
