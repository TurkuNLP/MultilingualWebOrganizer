#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/train_%j.out
#SBATCH -e ../../logs/train_%j.err

# Fine-tune mmBERT on the soft labels of a dataset made by create_datasets.sh.
# Usage (submit from this directory so relative log paths resolve):
#   sbatch train_classifier.sh <topic|format> <dataset_name> <run_name> [extra finetune.py args]
# e.g. sbatch train_classifier.sh topic dataset1 mmbert-cls
#      sbatch train_classifier.sh topic dataset1 mmbert-mean --classifier_pooling mean
#      sbatch --partition=gputest --time=00:15:00 train_classifier.sh topic dataset1 smoke --max_steps 20
#
# Extra arguments override the defaults below (later options win), e.g.
# --learning_rate 5e-5 or --num_train_epochs 5. The model is written to
# results/models/<label type>/<dataset_name>/<run_name>; rerunning the same command
# resumes from the last checkpoint there.

set -euo pipefail

usage="usage: sbatch train_classifier.sh <topic|format> <dataset_name> <run_name> [extra args]"
label_type="${1:?$usage}"
dataset_name="${2:?$usage}"
run_name="${3:?$usage}"
shift 3

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/finetune.py"
dataset_dir="$base_dir/results/datasets/$label_type/$dataset_name"
output_dir="$base_dir/results/models/$label_type/$dataset_name/$run_name"
echo "Dataset: $dataset_dir"
echo "Output:  $output_dir"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Effective batch size 32 (8 x 4 accumulation steps). Batches are grouped by length,
# so the longest batches hold 8 documents of up to 8,192 tokens; check gpu_mem_free
# in the first log lines and lower the per-device batch size if memory runs out.
python "$python_script" \
    --model_name jhu-clsp/mmBERT-base \
    --dataset_path "$dataset_dir" \
    --output_dir "$output_dir" \
    --do_train \
    --do_eval \
    --do_predict \
    --bf16 \
    --max_length 8192 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model js_divergence_bits \
    --greater_is_better false \
    --logging_steps 50 \
    --report_to none \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --seed 42 \
    "$@"
