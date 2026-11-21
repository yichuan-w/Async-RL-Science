# Export per-step curves of Countdown runs from their TensorBoard event files (one gzipped JSONL per run,
# <variant>__<method>.jsonl.gz, plus summary.csv). The released data in repro/data/runs was written by this script;
# every plot script in repro/ reads only these files.
#
#   python repro/export_runs.py [--runs RUNS_ROOT] [--out outputs/rerun_export]
#
# RUNS_ROOT holds one directory per run, RUNS_ROOT/<variant>/<method>/, each the dump_folder of a run started by
# repro/run.sh (the event files sit at its top level).
#
# Per step t (newest event file of a run wins on overlapping steps):
#   reward, val (every 50 steps), kl_k1, age_mean, age_max, groups_trained (64), groups_dropped (stale-dropped
#   that step), buffer_groups (completed groups held in the rollout buffer after the step), buffer_inflight (groups
#   still being generated),
#   prompt_tokens_mean / completion_tokens_mean of the trained batch, train_loss_tokens (completion tokens in the
#   loss), train_seq_tokens (prompt + completion of the trained samples).
# Cumulative compute (N = 0.6e9 parameters):
#   gen_groups_cum = sum_{s<=t} (groups_trained + groups_dropped) + buffer_groups(t): every group generated up to
#     the end of step t, i.e. trained, dropped, or still waiting in the buffer (this includes the prefill).
#   gen_prompt_tokens_cum / gen_completion_tokens_cum: accumulated once per group, when it enters the count: the groups
#     added at step t (gen_groups_cum(t) - gen_groups_cum(t-1), at least 0) x 8 samples x step t's trained-batch means.
#     Dropped and buffered groups are not logged token by token, so their lengths are approximated by the trained batch
#     of the step they were generated in.
#   train_tokens_cum = sum of train_seq_tokens (the trainer runs forward and backward over the whole sequence; the
#     loss covers only the completion tokens, train_loss_tokens).
#   gen_flops_cum = 2 N (gen_prompt_tokens_cum + gen_completion_tokens_cum); prefix caching (about half the prompt
#     tokens hit the cache) and the sampler's top-k logprobs are ignored.
#   train_flops_cum = 6 N train_tokens_cum (one forward + backward; this loss needs no extra reference forward).
#   flops_cum = gen_flops_cum + train_flops_cum.
# Wall clock:
#   wall_s: wall_time of the step's rollout_reward/_mean minus that of step 1, in seconds. Only the run's own event
#     files count: the newest file that starts at step 1 and any later (resumed) files; an older file starting at
#     step 1 belongs to an earlier run in the same directory and is skipped. Across a resume, the gap between the
#     last step before it and the first step after it is replaced by that step's own step_time_s, and the resumed
#     steps carry restart = 1 (0 otherwise). Steps redone after a resume keep the resumed run's values.
#   step_time_s (timing/step/total), fwd_bwd_s (timing/step/forward_backward), optim_s (timing/step/optim),
#     wait_batch_s (timing/step/wait_for_training_batch: the trainer waiting for generated groups).
#   gpu: H100 or H200 when the event file name says so, else unknown (wall times differ between GPU types).
import os, csv, glob, gzip, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(HERE, "..", "outputs", "countdown"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "rerun_export"))
args = ap.parse_args()
R, OUT = args.runs.rstrip("/") + "/", args.out
N, GROUP = 0.6e9, 8
TAGS = {
    "reward": "rollout_reward/_mean",
    "val": "validation_reward/_mean",
    "kl_k1": "is_grid/kl_samp_train",
    "age_mean": "train_batch/policy_age/mean",
    "age_max": "train_batch/policy_age_max",
    "groups_trained": "train_batch/num_rollout_groups",
    "groups_dropped": "rollout_buffer/num_groups_dropped_stale/sum",
    "buffer_finalized": "rollout_buffer/num_groups_finalized",
    "buffer_inflight": "rollout_buffer/num_groups_inflight",
    "prompt_tokens_mean": "rollout/prompt_length/mean",
    "completion_tokens_mean": "rollout/output_tokens/mean",
    "train_loss_tokens": "batch/num_global_valid_tokens",
    "samples": "train_batch/num_training_samples",
    "step_time_s": "timing/step/total/mean",
    "fwd_bwd_s": "timing/step/forward_backward/mean",
    "optim_s": "timing/step/optim/mean",
    "wait_batch_s": "timing/step/wait_for_training_batch/mean",
}
GPU = {"h100": "H100", "h200": "H200"}  # token in the event file name -> GPU type


def load(d):
    eas = []
    for f in sorted(glob.glob(d + "/events*")):  # file names start with the creation time
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        if "rollout_reward/_mean" in ea.Tags()["scalars"] and ea.Scalars("rollout_reward/_mean"):
            eas.append((f, ea))
    starts = [i for i, (_, ea) in enumerate(eas) if ea.Scalars("rollout_reward/_mean")[0].step == 1]
    eas = eas[starts[-1] :] if starts else eas  # drop earlier runs that share the directory
    s, wall, seg = {}, {}, {}
    for i, (f, ea) in enumerate(eas):
        have = ea.Tags()["scalars"]
        for k, t in TAGS.items():
            if t in have:
                s.setdefault(k, {}).update({e.step: e.value for e in ea.Scalars(t)})
        r = ea.Scalars("rollout_reward/_mean")
        for e in r:
            wall[e.step] = e.wall_time
            seg[e.step] = i
    parts = [x.rstrip("0123456789").lower() for x in os.path.basename(eas[-1][0]).split(".")] if eas else []
    s["gpu"] = next((GPU[x] for x in parts if x in GPU), "unknown")
    # wall_s with each resume gap replaced by the resumed step's own duration
    steps, out, off = sorted(wall), {}, 0.0
    for j, t in enumerate(steps):
        if j and seg[t] != seg[steps[j - 1]]:
            off += wall[t] - wall[steps[j - 1]] - s.get("step_time_s", {}).get(t, 0.0)
        out[t] = wall[t] - off - wall[steps[0]]
    s["wall_s"], s["restart"] = out, {t: int(i > 0) for t, i in seg.items()}
    s["segments"] = len(eas)
    return s


os.makedirs(OUT, exist_ok=True)
summary = []
for d in sorted(glob.glob(f"{R}*/*/")):
    v, m = d.rstrip("/").split("/")[-2:]
    s = load(d)
    if len(s.get("reward", {})) < 50:
        continue
    steps = sorted(s["reward"])
    g = lambda k, t, dflt=0.0: s.get(k, {}).get(t, dflt)
    cum = dict(groups=0.0, train_tokens=0.0, gp=0.0, gc=0.0)
    prev_gen = 0.0
    rows = []
    for t in steps:
        trained = g("groups_trained", t, 64.0)
        dropped = g("groups_dropped", t)
        buf = g("buffer_finalized", t)
        p, c = g("prompt_tokens_mean", t), g("completion_tokens_mean", t)
        nsamp = g("samples", t, trained * GROUP)
        cum["groups"] += trained + dropped
        cum["train_tokens"] += nsamp * (p + c)
        gen_groups = cum["groups"] + buf
        new = max(gen_groups - prev_gen, 0.0)
        prev_gen = max(prev_gen, gen_groups)
        cum["gp"] += new * GROUP * p
        cum["gc"] += new * GROUP * c
        gp, gc = cum["gp"], cum["gc"]
        row = dict(
            step=t,
            reward=s["reward"][t],
            val=s.get("val", {}).get(t),
            kl_k1=s.get("kl_k1", {}).get(t),
            age_mean=s.get("age_mean", {}).get(t),
            age_max=s.get("age_max", {}).get(t),
            groups_trained=trained,
            groups_dropped=dropped,
            buffer_groups=buf,
            buffer_inflight=s.get("buffer_inflight", {}).get(t),
            prompt_tokens_mean=p,
            completion_tokens_mean=c,
            train_loss_tokens=s.get("train_loss_tokens", {}).get(t),
            train_seq_tokens=nsamp * (p + c),
            gen_groups_cum=gen_groups,
            gen_prompt_tokens_cum=gp,
            gen_completion_tokens_cum=gc,
            train_tokens_cum=cum["train_tokens"],
            gen_flops_cum=2 * N * (gp + gc),
            train_flops_cum=6 * N * cum["train_tokens"],
        )
        row["flops_cum"] = row["gen_flops_cum"] + row["train_flops_cum"]
        row.update(
            wall_s=s["wall_s"].get(t),
            restart=s["restart"].get(t, 0),
            step_time_s=s.get("step_time_s", {}).get(t),
            fwd_bwd_s=s.get("fwd_bwd_s", {}).get(t),
            optim_s=s.get("optim_s", {}).get(t),
            wait_batch_s=s.get("wait_batch_s", {}).get(t),
            gpu=s["gpu"],
        )
        rows.append(row)

    def rnd(x):
        return float(f"{x:.6g}") if isinstance(x, float) else x

    with gzip.open(f"{OUT}/{v}__{m}.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps({k: rnd(x) for k, x in r.items()}) + "\n")
    last = rows[-1]
    summary.append(
        dict(
            variant=v,
            method=m,
            steps=last["step"],
            flops_cum=last["flops_cum"],
            gen_flops_frac=last["gen_flops_cum"] / last["flops_cum"],
            dropped_groups_cum=sum(r["groups_dropped"] for r in rows),
            gen_groups_cum=last["gen_groups_cum"],
            wall_s=last["wall_s"],
            restarts=s["segments"] - 1,
            gpu=last["gpu"],
        )
    )
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
print(OUT, len(summary), "runs")
