#!/usr/bin/env bash
#SBATCH --job-name=gemma_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=0-16:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err

set -euo pipefail
set -x

echo "======= Conda and CUDA ======="

module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate diffusiongemma-eval

echo "Python: $(which python)"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "NVCC: $(which nvcc || true)"
echo "================================"

# =========================================================
# DiffusionGemma 26B-A4B
# =========================================================

lm_eval \
    --model hf-diffusion-gemma \
    --model_args pretrained=google/diffusiongemma-26B-A4B-it,dtype=bfloat16,parallelize=True,max_length=8192,rm_self_attention=True \
    --tasks mmlu_pro \
    --batch_size 1 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path results/diffusiongemma_mmlu_pro_baseline_rm_sa