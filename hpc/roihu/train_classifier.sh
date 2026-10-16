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
# e.g. sbatch train_classifier.sh topic hplt-v1 mmbert-mean
#      sbatch train_classifier.sh topic hplt-v1 mmbert-cls --classifier_pooling cls
# (mmBERT-base pools by mean by default; --classifier_pooling cls uses the first token.)
#      TRAIN_DOCUMENTS=1000000 sbatch --time=24:00:00 train_classifier.sh topic hplt-v1 mmbert-1m
#      TRAIN_DOCUMENTS=320 EVALUATIONS=2 sbatch --partition=gputest --time=00:15:00 train_classifier.sh \
#          topic hplt-v1 smoke --max_eval_samples 500
#
# Training length: 3 epochs by default, evaluated and checkpointed after each epoch.
# TRAIN_DOCUMENTS=<n> trains on n documents instead: n / 32 optimizer steps at the
# batch size below, or n / 64 with --gradient_accumulation_steps 8, so the amount of
# data stays the same when the batch size changes. One pass over the training split is
# its size (in the dataset's README.md); larger n repeats documents. The run is
# evaluated and checkpointed EVALUATIONS times (default 10) at even intervals, the
# last at the final step. The best evaluation's checkpoint is kept either way. Large
# datasets also need a longer --time for sbatch.
#
# Extra arguments override the defaults below (later options win), e.g.
# --learning_rate 5e-5 or --num_train_epochs 5. The model is written to
# results/models/<label type>/<dataset_name>/<run_name>; rerunning the same command
# resumes from the last checkpoint there.
#
# The input follows the dataset: the URL, a blank line, and the text (as in
# WebOrganizer's default classifiers, URLs percent-decoded) if it has a url column,
# the text alone if it was made with create_datasets.sh ... --no_url. Override with
# --template. The template is saved in the model's config.json.

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

if [[ -n "${MAX_STEPS:-}${EVAL_STEPS:-}" ]]; then
    echo "MAX_STEPS and EVAL_STEPS were replaced by TRAIN_DOCUMENTS and EVALUATIONS" >&2
    exit 1
fi
if [[ -n "${TRAIN_DOCUMENTS:-}" ]]; then
    evaluations="${EVALUATIONS:-10}"
    if ! [[ "$TRAIN_DOCUMENTS" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAIN_DOCUMENTS must be a positive integer, got '$TRAIN_DOCUMENTS'" >&2
        exit 1
    fi
    # At least 2: an interval of 1 would mean every step instead of the whole run.
    if ! [[ "$evaluations" =~ ^[1-9][0-9]*$ ]] || (( evaluations < 2 )); then
        echo "EVALUATIONS must be an integer of at least 2, got '$evaluations'" >&2
        exit 1
    fi
    # The interval is a fraction of the run, because finetune.py derives the number of
    # steps from the batch size. The Trainer turns it into steps and also evaluates the
    # final step.
    interval=$(awk -v n="$evaluations" 'BEGIN { printf "%.6g", 1 / n }')
    # Checkpoints at every evaluation, as --load_best_model_at_end requires.
    schedule_args=(--max_train_documents "$TRAIN_DOCUMENTS" --eval_strategy steps --eval_steps "$interval"
                   --save_strategy steps --save_steps "$interval")
elif [[ -n "${EVALUATIONS:-}" ]]; then
    echo "EVALUATIONS needs TRAIN_DOCUMENTS; epoch-based runs evaluate every epoch" >&2
    exit 1
else
    schedule_args=(--num_train_epochs 3 --eval_strategy epoch --save_strategy epoch)
fi
echo "Schedule: ${schedule_args[*]}"

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
    --decode_url true \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 0.1 \
    "${schedule_args[@]}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
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
