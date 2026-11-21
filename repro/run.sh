#!/bin/bash
# Launch one Countdown run of the blog post on a single 8-GPU host.
#
#   bash repro/run.sh <variant> <method> [extra config overrides]
#   bash repro/run.sh async128randnp sc_klreg_kl03
#   bash repro/run.sh stale sc_kl03 --async_loop.num_training_steps 10    # quick check
#
# Layout used for every run in the post: a 2-GPU FSDP trainer plus 6 single-GPU vLLM
# samplers (8 GPUs). The run writes TensorBoard events, rollouts and checkpoints to
# ${OUT_ROOT:-outputs/countdown}/<variant>/<method>, the layout repro/export_runs.py reads.
# Set NUM_GENERATORS / TRAINER_GPUS for other GPU counts (fewer samplers make generation
# slower relative to training, which changes the batch policy age of the async variants),
# and RL_GPUS=0,1,... to pin the run to specific devices.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
    sed -n 2,14p "$0"; exit 1
fi
VARIANT=$1; METHOD=$2; shift 2
OUT="${OUT_ROOT:-outputs/countdown}/${VARIANT}/${METHOD}"
mkdir -p "${OUT}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Metrics go to TensorBoard (read by export_runs.py) and to W&B (project score-centering),
# offline unless WANDB_MODE is set (e.g. WANDB_MODE=online after `wandb login`).
export WANDB_MODE="${WANDB_MODE:-offline}"
exec python -m torchtitan.experiments.rl.train \
    --module countdown \
    --config "rl_countdown_qwen3_0_6b_${VARIANT}_${METHOD}" \
    --num_generators "${NUM_GENERATORS:-6}" \
    --trainer.parallelism.data_parallel_shard_degree "${TRAINER_GPUS:-2}" \
    --dump_folder "${OUT}" \
    "$@"
