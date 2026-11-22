# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Redraw the figures of "Beyond Score Centering: What Else Can We Optimize in Async RL?"
from the per-step run exports in ``repro/data/runs`` and the per-trajectory gradient
exports in ``repro/data/grad_by_age``.

    python repro/plot_blog.py                 # every figure -> repro/figures/
    python repro/plot_blog.py --fig 5 7       # selected figures
    python repro/plot_blog.py --data DIR      # exports written by export_runs.py elsewhere

Each figure function prints the numbers quoted in the post next to the ones it
computes, so a re-run can be checked against them. Figures 3 and 9 of the post are
schematics; Figure 3 is drawn here, Figure 9 (the trainer-sampler loop) is not.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TS = (200, 400, 600)
AGES = tuple(range(0, 65, 8))

# One color per method across all figures.
COLORS = {
    "recipe": "#7b3fbf",
    "sc_klreg": "#222222",
    "sc_kl03": "#1baf7a",
    "sc_kl01": "#0b7d55",
    "sc_kl1": "#8fd3b6",
    "tis_sc": "#eda100",
    "mis_sc": "#2a78d6",
    "dppo": "#d03b3b",
    "sc": "#eb6834",
    "pg": "#7a7a7a",
    "tis": "#b58a5a",
    "drift": "#e87ba4",
}
LABELS = {
    "recipe": "Recipe (SC + KL mask)",
    "sc_klreg": "SC + KL penalty",
    "sc_kl03": "SC + KL mask 0.3",
    "sc_kl01": "SC + KL mask 0.1",
    "sc_kl1": "SC + KL mask 1.0",
    "tis_sc": "TIS + SC",
    "mis_sc": "MIS + SC",
    "dppo": "DPPO",
    "sc": "SC",
    "pg": "PG",
    "tis": "TIS",
    "drift": "drift (PG - SC)",
}


def style(ax):
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def trailing(y, w=50):
    """Trailing mean over the last ``w`` points (fewer at the start)."""
    y = np.asarray(y, float)
    c = np.cumsum(np.insert(y, 0, 0.0))
    i = np.arange(1, len(y) + 1)
    return (c[i] - c[np.maximum(i - w, 0)]) / np.minimum(i, w)


class Runs:
    """Per-step exports, one ``<variant>__<method>.jsonl.gz`` per run."""

    def __init__(self, root: str):
        self.root = root
        self._cache: dict[tuple[str, str], dict | None] = {}

    def get(self, variant: str, method: str, max_step: int | None = None) -> dict | None:
        key = (variant, method)
        if key not in self._cache:
            path = os.path.join(self.root, f"{variant}__{method}.jsonl.gz")
            if not os.path.exists(path):
                self._cache[key] = None
            else:
                with gzip.open(path, "rt") as f:
                    rows = [json.loads(line) for line in f]
                cols = {k: np.array([r.get(k) for r in rows], dtype=object) for k in rows[0]}
                self._cache[key] = cols
        run = self._cache[key]
        if run is None or max_step is None:
            return run
        keep = np.array([s <= max_step for s in run["step"]])
        return {k: v[keep] for k, v in run.items()}

    @staticmethod
    def col(run: dict, key: str):
        """Steps and float values of ``key`` where it is logged."""
        ok = np.array([v is not None for v in run[key]])
        return run["step"][ok].astype(float), run[key][ok].astype(float)

    def last50(self, variant, method, n=None):
        run = self.get(variant, method, n)
        if run is None:
            return None
        _, y = self.col(run, "reward")
        return float(np.mean(y[-50:]))

    def first_reach(self, variant, method, level=0.6, n=None):
        run = self.get(variant, method, n)
        if run is None:
            return None
        x, y = self.col(run, "reward")
        hit = np.nonzero(trailing(y) >= level)[0]
        return int(x[hit[0]]) if len(hit) else None


def collapsed(v):
    return v is not None and v < 0.05


def fmt(v):
    return "missing" if v is None else ("collapsed" if collapsed(v) else f"{v:.3f}")


# ----------------------------------------------------------------------------- gradient profile


def load_grads(root):
    recs = []
    for t in TS:
        with open(os.path.join(root, f"grads_T{t}.jsonl")) as f:
            recs += [json.loads(line) for line in f]
    # Trajectories with zero advantage have no gradient and carry no norm fields.
    return [r for r in recs if "pg" in r]


def fig1(runs, grads, out):
    """Gradient profile by policy age: median norm, drift share, per-token KL."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=130)
    med = lambda key, a: np.median([r[key] for r in grads if r["age"] == a])  # noqa: E731
    for key in ("pg", "sc"):
        ax[0].plot(AGES, [1e3 * med(key, a) for a in AGES], "o-", color=COLORS[key], label=LABELS[key])
    share = [np.median([r["drift"] / r["pg"] for r in grads if r["age"] == a]) for a in AGES]
    ax[1].plot(AGES, share, "o-", color=COLORS["drift"])
    kl = [med("kl_k1", a) for a in AGES]
    ax[2].plot(AGES, kl, "o-", color="#444444")
    ax[2].set_yscale("log")
    ax[0].set_ylabel(r"median gradient norm ($\times 10^{-3}$)")
    ax[1].set_ylabel(r"drift share $\|g_{PG}-g_{SC}\|/\|g_{PG}\|$ (median)")
    ax[2].set_ylabel("per-token sampler-trainer KL (median)")
    ax[0].legend(frameon=False)
    for a in ax:
        a.set_xlabel("policy age of the trajectory")
        style(a)
    fig.suptitle("Figure 1. Gradient profile by policy age (Countdown, Qwen3-0.6B, steps 200/400/600 pooled)")
    fig.tight_layout()
    fig.savefig(out)
    print(f"  PG median {1e3 * med('pg', 0):.2f} -> {1e3 * med('pg', 64):.2f} (post: 1.56 -> 5.05)")
    print(f"  SC median -> {1e3 * med('sc', 64):.2f} (post: 2.40)")
    print(f"  drift share {share[0]:.2f} -> {share[-1]:.2f} (post: 0.09 -> 0.86)")
    print(f"  KL {kl[0]:.4f} -> {kl[-1]:.3f} (post: 0.0005 -> 0.104)")


def fig2(runs, grads, out):
    """Per-trajectory gradient-norm distributions by age: rows PG / SC / drift."""
    rows = ("pg", "sc", "drift")
    cols = [(f"step {t}", [r for r in grads if r["T"] == t]) for t in TS] + [("pooled", grads)]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(16, 8), dpi=130, sharex=True, sharey="row")
    for i, key in enumerate(rows):
        for j, (title, rs) in enumerate(cols):
            ax = axes[i, j]
            data = [[r[key] for r in rs if r["age"] == a] for a in AGES]
            ax.boxplot(
                data,
                positions=range(len(AGES)),
                widths=0.6,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor=COLORS[key], edgecolor="#52514e", linewidth=0.8),
                medianprops=dict(color="#0b0b0b", linewidth=1.2),
            )
            ax.set_yscale("log")
            ax.set_xticks(range(len(AGES)), [str(a) for a in AGES], fontsize=7)
            style(ax)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{LABELS[key]}\ngradient L2 norm")
            if i == len(rows) - 1:
                ax.set_xlabel("policy age")
    fig.suptitle(
        "Figure 2. Per-trajectory gradient norm by policy age (zero-advantage trajectories left out)"
    )
    fig.tight_layout()
    fig.savefig(out)


def fig3(runs, grads, out):
    """Schematic: the sum of a = 3 unit steps, orthogonal vs aligned."""
    fig = plt.figure(figsize=(9, 4), dpi=130)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for v, c in zip(np.eye(3), ("#2a78d6", "#1baf7a", "#eda100")):
        ax1.quiver(0, 0, 0, *v, color=c, arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 1, 1, 1, color="#222222", lw=2, arrow_length_ratio=0.08)
    ax1.set_title(r"orthogonal steps: $\|\sum v_i\| = \sqrt{a}$")
    ax1.set_xlim(0, 1.1), ax1.set_ylim(0, 1.1), ax1.set_zlim(0, 1.1)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_zticks([])
    ax2 = fig.add_subplot(1, 2, 2)
    for k, c in enumerate(("#2a78d6", "#1baf7a", "#eda100")):
        ax2.annotate("", xy=(k + 1, 0.1), xytext=(k, 0.1), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax2.annotate(
        "", xy=(3, -0.1), xytext=(0, -0.1), arrowprops=dict(arrowstyle="->", color="#222222", lw=2.5)
    )
    ax2.set_xlim(-0.2, 3.3), ax2.set_ylim(-0.5, 0.5), ax2.axis("off")
    ax2.set_title(r"aligned steps: $\|\sum v_i\| = a$")
    fig.suptitle("Figure 3. Summing a = 3 update steps")
    fig.tight_layout()
    fig.savefig(out)


def fig4(runs, grads, out):
    """Gradient profile with the KL mask: median norm relative to age 0."""
    keys = ("pg", "sc", "sc_kl1", "tis_sc", "sc_kl03", "sc_kl01")
    med = lambda key, a: np.median([r[key] for r in grads if r["age"] == a])  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130, gridspec_kw=dict(width_ratios=[1.4, 1]))
    ratios = {}
    for key in keys:
        base = med(key, 0)
        rel = [med(key, a) / base for a in AGES]
        ratios[key] = rel[-1]
        ax[0].plot(AGES, rel, "o-", color=COLORS[key], label=LABELS[key])
    masked = {
        k: np.mean([r[k + "_masked"] for r in grads if r["age"] == 64])
        for k in ("sc_kl1", "sc_kl03", "sc_kl01")
    }
    names = [LABELS[k] + (f"\n({100 * masked[k]:.1f}% masked)" if k in masked else "") for k in keys]
    ax[1].barh(range(len(keys)), [ratios[k] for k in keys], color=[COLORS[k] for k in keys])
    ax[1].set_yticks(range(len(keys)), names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].axvline(1.0, color="#999999", lw=0.8)
    for i, k in enumerate(keys):
        ax[1].text(ratios[k] + 0.03, i, f"{ratios[k]:.2f}", va="center", fontsize=8)
    ax[0].set_xlabel("policy age")
    ax[0].set_ylabel("median gradient norm / its value at age 0")
    ax[1].set_xlabel("age-64 / age-0 median norm")
    ax[0].legend(frameon=False, fontsize=8)
    for a in ax:
        style(a)
    fig.suptitle("Figure 4. Gradient profile with the KL mask")
    fig.tight_layout()
    fig.savefig(out)
    post = dict(pg=3.23, sc=1.54, sc_kl1=1.34, tis_sc=1.15, sc_kl03=1.06, sc_kl01=0.74)
    for k in keys:
        print(f"  {k}: age-64/age-0 {ratios[k]:.2f} (post: {post[k]})")


# ----------------------------------------------------------------------------- training runs


def fig5(runs, grads, out):
    """Final training accuracy at small staleness: staleness 64 and greedy async32."""
    methods = ("sc_klreg", "sc_kl03", "tis_sc", "sc", "pg", "tis")
    settings = (("stale", "staleness 64 (SC paper setting)"), ("async32", "greedy / FIFO, batch age ~31"))
    post = {
        "stale": dict(sc_klreg=0.710, sc_kl03=0.716, tis_sc=0.734, sc=0, pg=0, tis=0),
        "async32": dict(sc_klreg=0.720, sc_kl03=0.726, tis_sc=0.701, sc=0.486, pg=0, tis=0),
    }
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130, sharey=True)
    for a, (v, title) in zip(ax, settings):
        vals = [runs.last50(v, m, 1200) for m in methods]
        a.bar(range(len(methods)), [x or 0 for x in vals], color=[COLORS[m] for m in methods])
        for i, x in enumerate(vals):
            a.text(i, (x or 0) + 0.01, fmt(x), ha="center", fontsize=8)
        a.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=25, ha="right", fontsize=8)
        a.set_title(title)
        style(a)
        for m, x in zip(methods, vals):
            print(
                f"  {v}/{m}: last-50 train {fmt(x)} (post: {fmt(post[v][m]) if post[v][m] else 'collapsed'})"
            )
    ax[0].set_ylabel("training accuracy, mean of last 50 steps")
    fig.suptitle("Figure 5. Final training accuracy at small staleness (lr 1e-2, 1,200 steps, one seed)")
    fig.tight_layout()
    fig.savefig(out)


def _curves(ax_top, ax_bot, runs, variants, method, color, label, n):
    """Trailing-50 training accuracy and validation; band = range over seeds."""
    tr, va = [], []
    for v in variants:
        run = runs.get(v, method, n)
        if run is None:
            continue
        x, y = runs.col(run, "reward")
        tr.append((x, trailing(y)))
        xv, yv = runs.col(run, "val")
        va.append((xv, yv))
    if not tr:
        return
    for curves, ax, lw in ((tr, ax_top, 1.6), (va, ax_bot, 1.3)):
        m = min(len(c[0]) for c in curves)
        x = curves[0][0][:m]
        ys = np.stack([c[1][:m] for c in curves])
        ax.plot(
            x,
            ys.mean(0),
            color=color,
            lw=lw,
            label=label if ax is ax_top else None,
            marker="o" if ax is ax_bot else None,
            ms=2,
        )
        if len(curves) > 1:
            ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.18, lw=0)


def fig6(runs, grads, out):
    """SC + KL mask vs TIS + SC and MIS + SC: 128-step buffer, age 127, 3 seeds, 2,400 steps."""
    seeds = ("async128s1", "async128s2", "async128s3")
    methods = ("sc_kl03", "tis_sc", "mis_sc")
    fig, ax = plt.subplots(2, 3, figsize=(16, 7), dpi=130, sharex=True, sharey="row")
    for j, m in enumerate(methods):
        for k, v in enumerate(seeds):
            run = runs.get(v, m, 2400)
            if run is None:
                continue
            x, y = runs.col(run, "reward")
            ax[0, j].plot(
                x, trailing(y), color=COLORS[m], lw=1.3, alpha=0.5 + 0.25 * k, label=f"seed {k + 1}"
            )
            xv, yv = runs.col(run, "val")
            ax[1, j].plot(xv, yv, color=COLORS[m], lw=1.1, alpha=0.5 + 0.25 * k, marker="o", ms=2)
            print(f"  {v}/{m}: last-50 train {fmt(runs.last50(v, m, 2400))}")
        ax[0, j].set_title(LABELS[m])
        ax[0, j].legend(frameon=False, fontsize=8)
    ax[0, 0].set_ylabel("training accuracy (trailing 50)")
    ax[1, 0].set_ylabel("validation accuracy")
    for a in ax.flat:
        style(a)
    for a in ax[1]:
        a.set_xlabel("training step")
    fig.suptitle(
        "Figure 6. SC + KL mask vs TIS + SC and MIS + SC (128-step buffer, every batch at age 127, 3 seeds)"
    )
    fig.tight_layout()
    fig.savefig(out)


# Figure 7 / 8: the recipe against the baselines, 128-step buffer, no prefill.
RECIPE_COLUMNS = (
    ("greedy, lr 1e-2, tau 0.3", ("async128np",), "sc_klreg_kl03"),
    ("random, lr 1e-2, tau 0.3", ("async128randnp",), "sc_klreg_kl03"),
    (
        "random, lr 4e-2, tau 0.1",
        ("async128randnplr4", "async128randnplr4s1", "async128randnplr4s2"),
        "sc_klreg_kl01",
    ),
)
RECIPE_METHODS = ("recipe", "sc_klreg", "tis_sc", "mis_sc", "dppo")


def fig7(runs, grads, out):
    fig, ax = plt.subplots(2, 3, figsize=(17, 7.5), dpi=130, sharex=True, sharey="row")
    for j, (title, variants, recipe) in enumerate(RECIPE_COLUMNS):
        for m in RECIPE_METHODS:
            key = recipe if m == "recipe" else m
            _curves(ax[0, j], ax[1, j], runs, variants, key, COLORS[m], LABELS[m], 1200)
        ax[0, j].set_title(title)
        ax[0, j].set_ylim(0, 0.85)
        ax[1, j].set_ylim(0, 0.85)
    ax[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0, 0].set_ylabel("training accuracy (trailing 50)")
    ax[1, 0].set_ylabel("validation accuracy")
    for a in ax.flat:
        style(a)
    for a in ax[1]:
        a.set_xlabel("training step")
    fig.suptitle("Figure 7. The recipe against the baselines, 128-step buffer (band = range over seeds)")
    fig.tight_layout()
    fig.savefig(out)


def fig8(runs, grads, out):
    post = {
        "recipe": ("0.688", "0.707 (0.703 at tau 0.1)", "0.735 over seeds 0-1"),
        "sc_klreg": ("0.690", "0.716", "0.751"),
        "tis_sc": ("0.593", "0.691", "unstable"),
        "mis_sc": ("0.666", "0.680", "collapsed"),
        "dppo": ("collapsed", "0.700", "collapsed"),
    }
    fig, ax = plt.subplots(2, 3, figsize=(16, 7), dpi=130, sharey="row")
    for j, (title, variants, recipe) in enumerate(RECIPE_COLUMNS):
        finals, reach = [], []
        for m in RECIPE_METHODS:
            key = recipe if m == "recipe" else m
            f = [runs.last50(v, key, 1200) for v in variants]
            f = [x for x in f if x is not None]
            r = [runs.first_reach(v, key, 0.6, 1200) for v in variants if runs.get(v, key) is not None]
            finals.append(float(np.mean(f)) if f else None)
            reach.append(float(np.mean(r)) if r and all(x is not None for x in r) else None)
            print(
                f"  {title} / {m}: last-50 {fmt(finals[-1])} over {len(f)} seed(s) (post: {post[m][j]}),"
                f" reaches 0.6 at {'never' if reach[-1] is None else f'{reach[-1]:.0f}'}"
            )
        xs = range(len(RECIPE_METHODS))
        ax[0, j].bar(xs, [x or 0 for x in finals], color=[COLORS[m] for m in RECIPE_METHODS])
        ax[1, j].bar(xs, [x or 0 for x in reach], color=[COLORS[m] for m in RECIPE_METHODS])
        for i, (f, r) in enumerate(zip(finals, reach)):
            ax[0, j].text(i, (f or 0) + 0.01, fmt(f), ha="center", fontsize=8)
            ax[1, j].text(i, (r or 0) + 10, "never" if r is None else f"{r:.0f}", ha="center", fontsize=8)
        for a in ax[:, j]:
            a.set_xticks(
                xs, [LABELS[m].split(" (")[0] for m in RECIPE_METHODS], rotation=25, ha="right", fontsize=8
            )
            style(a)
        ax[0, j].set_title(title)
    ax[0, 0].set_ylabel("training accuracy, mean of last 50 steps")
    ax[1, 0].set_ylabel("first step with trailing-50 mean >= 0.6")
    fig.suptitle("Figure 8. Summary of Figure 7 (third column: mean over seeds)")
    fig.tight_layout()
    fig.savefig(out)


def fig10(runs, grads, out):
    """Batch policy age under each take strategy."""
    panels = (
        ("staleness 64", "stale", "sc_klreg"),
        ("greedy, K = 128", "async128np", "sc_klreg"),
        ("LIFO, K = 128", "async128lifonp", "sc_klreg"),
        ("random, K = 128", "async128randnp", "sc_klreg"),
    )
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), dpi=130, sharey=True)
    for a, (title, v, m) in zip(ax, panels):
        run = runs.get(v, m, 1200)
        if run is None:
            continue
        x, y = runs.col(run, "age_mean")
        if v == "async128randnp":
            a.plot(x, trailing(y, 10), color="#1baf7a", lw=1.3, label="mean (10-step smoothing)")
            xm, ym = runs.col(run, "age_max")
            a.plot(xm, ym, color="#1baf7a", lw=0.8, alpha=0.5, label="max")
            a.legend(frameon=False, fontsize=8)
            print(
                f"  random: mean age after step 200 {np.mean(y[x > 200]):.1f} (post: ~52), max {np.max(ym):.0f}"
            )
        else:
            a.plot(x, y, color="#222222", lw=1.2)
            print(f"  {title}: max {np.max(y):.0f}, median after step 200 {np.median(y[x > 200]):.1f}")
        a.set_title(title)
        a.set_xlabel("training step")
        style(a)
    ax[0].set_ylabel("batch policy age (mean)")
    fig.suptitle("Figure 10. Policy age of the training batch (K = 128 panels: SC + KL penalty runs)")
    fig.tight_layout()
    fig.savefig(out)


def fig11(runs, grads, out):
    """Greedy vs random take from a 128-step buffer, against training time."""
    # (run method, style key): the first panel is the recipe, SC + KL mask 0.3.
    methods = (("sc_klreg_kl03", "recipe"), ("tis_sc", "tis_sc"), ("mis_sc", "mis_sc"), ("dppo", "dppo"))
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), dpi=130, sharey="row")
    for j, (m, key) in enumerate(methods):
        for v, tag, ls in (("async128np", "greedy", "--"), ("async128randnp", "random", "-")):
            run = runs.get(v, m, 1200)
            if run is None:
                continue
            ok = np.array([w is not None and r is not None for w, r in zip(run["wall_s"], run["reward"])])
            h = run["wall_s"][ok].astype(float) / 3600
            r = run["reward"][ok].astype(float)
            ax[0, j].plot(h, r, color=COLORS[key], lw=0.4, alpha=0.25, ls=ls)
            ax[0, j].plot(h, trailing(r), color=COLORS[key], lw=1.8, ls=ls, label=tag)
            steps, wall = run["step"], run["wall_s"]
            vt = {s: w for s, w in zip(steps, wall) if w is not None}
            xv, yv = runs.col(run, "val")
            ax[1, j].plot(
                [vt.get(s, np.nan) / 3600 for s in xv], yv, color=COLORS[key], ls=ls, marker="o", ms=2
            )
            print(
                f"  {v}/{m}: {h[-1]:.2f} h for {int(steps[-1])} steps, last-50 {fmt(runs.last50(v, m, 1200))}"
            )
        ax[0, j].set_title("SC + KL mask" if key == "recipe" else LABELS[key])
        ax[0, j].legend(frameon=False, fontsize=8)
    ax[0, 0].set_ylabel("training accuracy")
    ax[1, 0].set_ylabel("validation accuracy")
    for a in ax.flat:
        style(a)
    for a in ax[1]:
        a.set_xlabel("training time (hours)")
    fig.suptitle(
        "Figure 11. Greedy (dashed) vs random (solid) take, 128-step buffer, no prefill"
        " (some runs trained on H200, the rest on H100; see the gpu field)"
    )
    fig.tight_layout()
    fig.savefig(out)


FIGS = {
    1: ("fig01_grad_profile.png", fig1, True),
    2: ("fig02_grad_by_age.png", fig2, True),
    3: ("fig03_step_sum.png", fig3, False),
    4: ("fig04_mask_profile.png", fig4, True),
    5: ("fig05_small_staleness.png", fig5, False),
    6: ("fig06_mask_vs_is.png", fig6, False),
    7: ("fig07_recipe.png", fig7, False),
    8: ("fig08_recipe_bars.png", fig8, False),
    10: ("fig10_batch_age.png", fig10, False),
    11: ("fig11_greedy_vs_random_time.png", fig11, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, nargs="*", default=sorted(FIGS))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "runs"))
    ap.add_argument("--grads", default=os.path.join(HERE, "data", "grad_by_age"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = Runs(args.data)
    grads = None
    for n in args.fig:
        name, fn, needs_grads = FIGS[n]
        if needs_grads and grads is None:
            grads = load_grads(args.grads)
        print(f"Figure {n} -> {name}")
        fn(runs, grads, os.path.join(args.out, name))
        plt.close("all")


if __name__ == "__main__":
    main()
