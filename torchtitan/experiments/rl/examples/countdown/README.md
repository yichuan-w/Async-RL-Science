# Countdown

Qwen3-0.6B on Countdown (Gandhi et al., 2024), the small-model benchmark of
[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)
(Marek & Ryabinin, 2026). Reference implementation:
[martin-marek/score-centering](https://github.com/martin-marek/score-centering).

## Environment

Single turn. Four numbers in 1..100 and a target up to 1000; the model has to
write an expression in `<answer> </answer>` tags that uses each number once with
`+ - * /` and equals the target. Reward 1 or 0. Problems are generated exactly as
in the reference (`envs/countdown.py`, seed 0, 20,000 problems); the first 64
after the seed-0 shuffle are held out for validation.

## Recipe (paper Appendix B)

| | |
|---|---|
| objective | REINFORCE, +1/-1 group-centered advantages (`AdvantageEstimator(scale=2)` on 0/1 rewards) |
| optimizer | plain SGD, lr 1e-2, no momentum, no clipping, one step per batch |
| batch | 64 prompts x 8 completions, zero-variance groups kept |
| length | 512 tokens (prompt ~133 + response <= 384) |
| sampling | temperature 1, top-p 1, chat template with thinking off |
| loss normalization | sum over tokens / all completion tokens of the step |
| correction | one point of Table 1 via `ISGridLoss` (`losses/is_grid.py`) |

`ISGridLoss` (`losses/is_grid.py`) ports the reference `losses.py`: every method is
REINFORCE with a detached weight `w(p/q)` from one grid (`level`, `stat`, `low`,
`high`, `outside`, `sign`); `score_centering=True` subtracts the expected score
under `q * w` using the sampler's top-k logprobs (`SamplingConfig.num_topk_logprobs`,
k = 128) with the tail modeled as `rho * p`. On top of that grid the same loss
implements the two additions studied in the blog post:

- **KL penalty** (`kl_coef`, `kl_age_schedule`): `- beta_a * KL(mu || pi)` per token,
  `beta_a = kl_coef / sqrt(a)` for a trajectory of policy age `a`.
- **KL mask** (`kl_mask_threshold`): drop every position whose head
  `KL(mu(.|s) || pi(.|s))` exceeds the threshold, from both the policy-gradient
  and the KL-penalty term.

The method names (`NAMED_METHODS`) used in the post are `pg`, `tis`, `sc`,
`tis_sc`, `mis_sc`, `dppo`, `sc_kl03` / `sc_kl01` / `sc_kl1` (SC + KL mask),
`sc_klreg` (SC + KL penalty) and `sc_klreg_kl03` / `sc_klreg_kl01` (SC + KL penalty
+ KL mask, the recipe).

## Async settings

Configs are named `rl_countdown_qwen3_0_6b_<variant>_<method>`
(`config_registry.py`). The variants used in the post:

| variant | sampler weights | buffer take | policy age of a batch |
|---|---|---|---|
| `stale` | refreshed every 65 steps (`weight_sync_interval = 65`) | oldest first | sawtooth 0..64 (the SC paper's "staleness 64") |
| `async32` | synced every step, 32-step buffer | oldest first (greedy / FIFO) | ~31 |
| `async128np` | synced every step, 128-step buffer, no prefill | oldest first | climbs to 127 |
| `async128randnp` | same | uniform at random (`sample_order="random"`) | mean ~52, max ~127 |
| `async128lifonp` | same | newest first (`sample_order="lifo"`) | ~3 after the initial backlog |
| `async128s{1,2,3}` | 128-step buffer prefilled before training, 2,400 steps, seeds 1-3 | oldest first | 127 throughout |
| `async128randnplr4[s1,s2]` | as `async128randnp`, lr 4e-2, seeds 0-2 | random | mean ~52 |

With a deep buffer the age of a batch is set by how far generation runs ahead of
training. The post's runs use six single-GPU vLLM samplers per trainer, which
keeps the buffer full; with fewer samplers the async variants see younger data.
`train_batch/policy_age/*` counts from admission into the buffer.

## Running

See `repro/README.md` at the repository root for the full list of runs behind each
figure of the post, the launcher (`repro/run.sh`) and the plotting scripts.

```bash
bash repro/run.sh stale sc_kl03            # one 8-GPU host
```

`VLLM_USE_FLASHINFER_SAMPLER=0` (set by `repro/run.sh`) avoids a flashinfer JIT link
failure in some environments.
