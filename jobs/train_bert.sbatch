#!/bin/bash
#SBATCH --job-name=trm_bert_e1_5ep
#SBATCH --partition=rocky
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:2
#SBATCH --output=/home/aburlova/trm/paper/slurm_logs/%x_%j.out
#SBATCH --error=/home/aburlova/trm/paper/slurm_logs/%x_%j.err

set -euo pipefail

module purge
module load Python/Miniconda CUDA/12.4
source activate /home/aburlova/conda_envs/mfb

# Compute-узлы работают без интернета. Все необходимые данные,
# tokenizer и веса BERT находятся внутри проекта.
export HF_HOME=/home/aburlova/trm/paper/.cache/huggingface
export HF_HUB_CACHE=/home/aburlova/trm/paper/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/home/aburlova/trm/paper/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Два DDP-процесса делят выделенные четыре CPU.
export OMP_NUM_THREADS=2

cd /home/aburlova/trm/paper

torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/base.yaml \
           configs/variants/bert_finetune.yaml \
           configs/local.yaml
