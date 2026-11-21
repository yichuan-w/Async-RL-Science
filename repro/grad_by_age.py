# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-trajectory gradient norm against policy age: the data behind Figures 1, 2 and 4 of the post.

Does an older trajectory move the update more without score centering, and does SC even that out?

Weights: one staleness-64 TIS + SC run to 600 steps with a checkpoint every 8 steps:

    bash repro/run.sh stale tis_sc --async_loop.num_training_steps 600 \
        --trainer.checkpoint.interval 8 --trainer.checkpoint.keep_latest_k 0

For each T in TS a mixed batch is built offline: the 64 prompts training step T reads, 8 samples each from every
theta_{T-a}, a in AGES (vLLM, temperature 1, 384 tokens, top-128 logprobs, as the sampler generates), advantage
2 (r - group mean). Each trajectory's loss under every method in METHODS is taken at theta_T (fp32 weights, bf16
autocast compute like the trainer's mixed precision), divided by the whole batch's completion tokens, and its
parameter-gradient L2 norm recorded (no clipping). "drift" is the gradient of SC's centering term alone
(SC - PG); "drift_tis" the same for TIS + SC - TIS.

    python repro/grad_by_age.py convert              # DCP checkpoints -> HF fp32, WORK/hf
    python repro/grad_by_age.py sample <gpu> <i> <n> # vLLM over every n-th checkpoint from i -> WORK/samples
    python repro/grad_by_age.py grads <gpu> <T>      # -> WORK/grads_T<T>.jsonl, one line per trajectory
    python repro/plot_blog.py --fig 1 2 4 --grads WORK

CKPT (default outputs/countdown/stale/tis_sc/checkpoint) holds the step-N checkpoint folders; WORK (default
outputs/grad_by_age) receives everything this script writes. The post's own output is in repro/data/grad_by_age.
"""

import glob, json, math, os, shutil, sys
import numpy as np, torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO)
WORK = os.environ.get("GRADAGE_WORK", os.path.join(REPO, "outputs/grad_by_age"))
CKPT = os.environ.get("GRADAGE_CKPT", os.path.join(REPO, "outputs/countdown/stale/tis_sc/checkpoint"))
HF_ASSETS = REPO + "/torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B"
TS = tuple(int(t) for t in os.environ.get("GRADAGE_TS", "200,400,600").split(","))
AGES = tuple(range(0, 65, 8))
NUM_PROMPTS, GROUP, BATCH_PROMPTS, TOPK, MAX_TOKENS = 64, 8, 64, 128, 384
METHODS = ("pg", "sc", "tis", "tis_sc", "sc_kl01", "sc_kl03", "sc_kl1")
STEPS = sorted({t - a for t in TS for a in AGES})


def convert():
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file
    from torchtitan.models.qwen3 import model_registry

    spec = model_registry("0.6B")
    adapter = spec.state_dict_adapter(spec.model, HF_ASSETS)
    for s in STEPS:
        dst = f"{WORK}/hf/step-{s}"
        if os.path.exists(dst + "/model.safetensors"):
            continue
        src = glob.glob(f"{CKPT}/step-{s}/**/.metadata", recursive=True)[0].rsplit("/", 1)[0]
        md = dcp.FileSystemReader(src).read_metadata()
        sd = {
            k: torch.empty(m.size, dtype=m.properties.dtype)
            for k, m in md.state_dict_metadata.items()
            if not k.startswith(("optimizer", "train_state", "lr_scheduler")) and hasattr(m, "size")
        }
        dcp.load(sd, checkpoint_id=src)
        hf = {k: v.contiguous() for k, v in adapter.to_hf(sd).items()}
        os.makedirs(dst, exist_ok=True)
        save_file(hf, dst + "/model.safetensors", metadata={"format": "pt"})
        for f in os.listdir(HF_ASSETS):
            if f.endswith(".json") and "index" not in f or f.endswith(".txt"):
                shutil.copy(f"{HF_ASSETS}/{f}", dst)
        print("converted", s, len(hf), flush=True)


def problems(t):
    from torchtitan.experiments.rl.examples.countdown.data import CountdownDataset, _problems

    c = CountdownDataset.Config()
    rest = _problems(c)[c.num_validation :]
    return [rest[(BATCH_PROMPTS * t + i) % len(rest)] for i in range(NUM_PROMPTS)]


def sample(gpu, i, n):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from torchtitan.experiments.rl.examples.countdown.data import SYSTEM_PROMPT, instruction
    from torchtitan.experiments.rl.examples.countdown.rubric import score_countdown
    from torchtitan.experiments.rl.renderer import RendererConfig

    renderer = RendererConfig(name="qwen3", enable_thinking=False).build(tokenizer_path=HF_ASSETS)
    tok = AutoTokenizer.from_pretrained(HF_ASSETS)
    stop = list(renderer.get_stop_token_ids())
    os.makedirs(f"{WORK}/samples", exist_ok=True)
    for s in STEPS[i::n]:
        jobs = [
            (t, t - s) for t in TS if t - s in AGES and not os.path.exists(f"{WORK}/samples/T{t}_a{t - s}.pt")
        ]
        if not jobs:
            continue
        llm = LLM(
            model=f"{WORK}/hf/step-{s}",
            dtype="bfloat16",
            gpu_memory_utilization=0.6,
            max_model_len=1024,
            max_logprobs=TOPK,
            seed=s,
            enforce_eager=True,
        )
        for t, a in jobs:
            probs = problems(t)
            prompts = [
                renderer.render_ids(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction(list(p.numbers), p.target)},
                    ],
                    tools=None,
                    add_generation_prompt=True,
                )
                for p in probs
            ]
            sp = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=MAX_TOKENS,
                n=GROUP,
                seed=1000 * t + a,
                stop_token_ids=stop,
                logprobs=TOPK,
                flat_logprobs=True,
                detokenize=False,
            )
            outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
            rows = []
            for pi, (p, o) in enumerate(zip(probs, outs)):
                for c in o.outputs:
                    f = c.logprobs
                    w = f.end_indices[0] - f.start_indices[0]
                    ids = torch.tensor(f.token_ids, dtype=torch.int32).view(-1, w)
                    lps = torch.tensor(f.logprobs, dtype=torch.float32).view(-1, w)
                    r = score_countdown(tok.decode(c.token_ids), p.numbers, p.target)
                    rows.append(
                        dict(
                            prompt=pi,
                            prompt_ids=list(o.prompt_token_ids),
                            ids=list(c.token_ids),
                            logq=lps[:, 0],
                            topk_ids=ids[:, 1:],
                            topk_lp=lps[:, 1:],
                            reward=r,
                        )
                    )
            torch.save(rows, f"{WORK}/samples/T{t}_a{a}.pt")
            print(f'sampled T{t} a{a} acc {np.mean([r["reward"] for r in rows]):.3f}', flush=True)
        del llm
        torch.cuda.empty_cache()


def grads(gpu, t):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from transformers import AutoModelForCausalLM
    from torchtitan.experiments.rl.losses.is_grid import ISGridLoss, NAMED_METHODS

    losses = {m: ISGridLoss(ISGridLoss.Config(**NAMED_METHODS[m])) for m in METHODS}
    model = AutoModelForCausalLM.from_pretrained(f"{WORK}/hf/step-{t}", torch_dtype=torch.float32).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    batch = []
    for a in AGES:
        rows = torch.load(f"{WORK}/samples/T{t}_a{a}.pt")
        for pi in range(NUM_PROMPTS):
            g = [r for r in rows if r["prompt"] == pi]
            mean = np.mean([r["reward"] for r in g])
            batch += [dict(r, age=a, adv=2.0 * (r["reward"] - mean)) for r in g]
    denom = sum(len(r["ids"]) for r in batch)
    out = open(f"{WORK}/grads_T{t}.jsonl", "w")
    for k, r in enumerate(batch):
        rec = dict(
            T=t, age=r["age"], prompt=r["prompt"], reward=r["reward"], adv=r["adv"], tokens=len(r["ids"])
        )
        if r["adv"] != 0:
            ids = torch.tensor(r["prompt_ids"] + r["ids"], device="cuda")[None]
            n_p, n_c = len(r["prompt_ids"]), len(r["ids"])
            with torch.autocast("cuda", dtype=torch.bfloat16):  # the trainer's bf16 mixed precision
                logits = model(ids).logits[:, n_p - 1 : -1].float()
            labels = ids[:, n_p:]
            kw = dict(
                generator_logprobs=r["logq"][None].cuda(),
                advantages=torch.full((1, n_c), r["adv"], device="cuda"),
                loss_mask=torch.ones(1, n_c, dtype=torch.bool, device="cuda"),
                topk_token_ids=r["topk_ids"][None].cuda(),
                topk_logprobs=r["topk_lp"][None].cuda(),
                positions=torch.arange(n_c, device="cuda")[None],
            )
            g = {}
            for m in METHODS:
                loss, met = losses[m](logits, labels, denom, **kw)
                g[m] = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                rec[m] = math.sqrt(sum(float(x.double().square().sum()) for x in g[m] if x is not None))
                if m.startswith("sc_kl"):
                    rec[m + "_masked"] = float(met["is_grid/kl_masked_frac"]) * denom / n_c
                if m == "tis_sc":
                    rec["head_tv"] = float(met["sc/head_tv_mean"]) * denom / n_c
                    rec["kl_k1"] = float(met["is_grid/kl_samp_train"]) * denom / n_c
            for name, (x, y) in {"drift": ("sc", "pg"), "drift_tis": ("tis_sc", "tis")}.items():
                rec[name] = math.sqrt(
                    sum(float((u - v).double().square().sum()) for u, v in zip(g[x], g[y]) if u is not None)
                )
            del g, logits
        out.write(json.dumps(rec) + "\n")
        if k % 100 == 0:
            print(t, k, len(batch), flush=True)
    out.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "convert":
        convert()
    elif cmd == "sample":
        sample(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "grads":
        grads(sys.argv[2], int(sys.argv[3]))
    else:
        sys.exit(__doc__)
