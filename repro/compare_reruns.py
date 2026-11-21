# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlay re-runs on the released runs: training accuracy (trailing 50), validation
and batch policy age for every <variant>__<method> present in both export folders.

    python repro/export_runs.py --runs outputs/countdown --out outputs/rerun_export
    python repro/compare_reruns.py --rerun outputs/rerun_export     # -> repro/figures/reruns.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_blog import HERE, Runs, fmt, style, trailing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", required=True, help="export folder of the re-runs")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "reruns.png"))
    args = ap.parse_args()
    released, rerun = Runs(args.data), Runs(args.rerun)
    names = sorted(
        f[: -len(".jsonl.gz")]
        for f in os.listdir(args.rerun)
        if f.endswith(".jsonl.gz") and os.path.exists(os.path.join(args.data, f))
    )
    if not names:
        raise SystemExit("no run is present in both folders")
    fig, ax = plt.subplots(len(names), 3, figsize=(16, 3.2 * len(names)), dpi=110, squeeze=False)
    print(f"{'run':40s} {'released':>10s} {'re-run':>10s}  (last-50 training accuracy; final validation)")
    for i, name in enumerate(names):
        v, m = name.split("__")
        n = max(released.get(v, m)["step"])
        for runs, tag, ls in ((released, "released", "-"), (rerun, "re-run", "--")):
            run = runs.get(v, m, n)
            x, y = runs.col(run, "reward")
            ax[i, 0].plot(x, trailing(y), ls=ls, lw=1.5, label=f"{tag} ({int(x[-1])} steps)")
            xv, yv = runs.col(run, "val")
            ax[i, 1].plot(xv, yv, ls=ls, lw=1.2, marker="o", ms=2)
            xa, ya = runs.col(run, "age_mean")
            ax[i, 2].plot(xa, ya, ls=ls, lw=1.0)
        a, b = released.get(v, m, n), rerun.get(v, m, n)
        va, vb = released.col(a, "val")[1], rerun.col(b, "val")[1]
        print(
            f"{name:40s} {fmt(released.last50(v, m, n)):>10s} {fmt(rerun.last50(v, m, n)):>10s}"
            f"   val {va[-1]:.3f} / {vb[-1]:.3f}"
        )
        ax[i, 0].set_ylabel(f"{v}\n{m}\ntraining acc. (trailing 50)", fontsize=8)
        ax[i, 0].legend(frameon=False, fontsize=8)
        ax[i, 1].set_ylabel("validation accuracy", fontsize=8)
        ax[i, 2].set_ylabel("batch policy age", fontsize=8)
        for a_ in ax[i]:
            style(a_)
    for a_ in ax[-1]:
        a_.set_xlabel("training step")
    fig.suptitle("Released runs (solid) vs re-runs from this repository (dashed)")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
