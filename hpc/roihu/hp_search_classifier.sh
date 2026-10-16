#!/bin/bash
#SBATCH --job-name=hp_search_classifier
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/hp_search_%A_%a.out
#SBATCH -e ../../logs/hp_search_%A_%a.err

# One trial of a random hyperparameter search per array task (trial = array task ID).
# Usage (submit from this directory so relative log paths resolve):
#   sbatch --array=0-15%4 hp_search_classifier.sh <topic|format> <dataset_name> <search_name> [extra finetune.py args]
# e.g. sbatch --array=0-15%4 hp_search_classifier.sh topic hplt-v1 search1
#      sbatch --array=0-15%4 hp_search_classifier.sh topic hplt-v1 search1-50k --max_train_samples 50000
#      TRAIN_DOCUMENTS=144000 sbatch --array=0-15%4 hp_search_classifier.sh topic hplt-v1 docs144k \
#          --max_eval_samples 5000
# Smoke test first (one short trial, its own search name so it doesn't mix with real trials):
#      sbatch --array=0 --partition=gputest --time=00:15:00 hp_search_classifier.sh topic hplt-v1 smoke \
#          --max_train_samples 2000 --max_eval_samples 500
#      TRAIN_DOCUMENTS=2000 EVALUATIONS=2 sbatch --array=0 --partition=gputest --time=00:15:00 \
#          hp_search_classifier.sh topic hplt-v1 smoke-docs --max_train_samples 2000 --max_eval_samples 500
#
# %4 runs at most four trials at a time. Each trial trains on the training split (or a
# fixed random subset with --max_train_samples), evaluates on the validation split
# every epoch, and is scored by its best js_divergence_bits. Trials save no
# checkpoints and never touch the test split.
#
# Document-budget trials: with TRAIN_DOCUMENTS=<n> every trial trains on n documents,
# in n / batch size optimizer steps, and evaluates EVALUATIONS times (default 10). The
# default search space is then hp_space_budget.yaml, which searches the batch size but
# not epochs: all trials see the same data, so a larger batch can't win by training
# longer. On a large validation split, add --max_eval_samples to keep the evaluations
# cheap.
#
# The input follows the dataset, as in train_classifier.sh: URL and text if it has a
# url column, the text alone if it was made with --no_url. Each trial records its
# template, and summarize_hp_search.py refuses to rank trials with different inputs.
#
# Search space: configs/roihu/classifier/hp_space.yaml, or hp_space_budget.yaml with
# TRAIN_DOCUMENTS (override either with HP_SPACE=<path>).
# Trial i always gets the same hyperparameters for the same space and --hp_search_seed,
# so more trials can be added later (e.g. --array=16-31) and failed ones rerun with the
# same command; completed trials are skipped.
#
# Results: results/models/<label type>/<dataset_name>/hp_search/<search_name>/trial-*/trial.json
# Rank them, and get the flags for train_classifier.sh, on a Roihu-GPU login node with:
#   module load python-vllm/0.19.1
#   python ../../scripts/model_training/summarize_hp_search.py \
#       ../../results/models/<label type>/<dataset_name>/hp_search/<search_name>

set -euo pipefail

usage="usage: sbatch --array=0-15%4 hp_search_classifier.sh <topic|format> <dataset_name> <search_name> [extra args]"
label_type="${1:?$usage}"
dataset_name="${2:?$usage}"
search_name="${3:?$usage}"
shift 3
trial="${SLURM_ARRAY_TASK_ID:?submit as an array job, e.g. sbatch --array=0-15%4 ...}"

echo "Job ${SLURM_ARRAY_JOB_ID:-}_${trial} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/finetune.py"
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
    schedule_args=(--max_train_documents "$TRAIN_DOCUMENTS" --eval_strategy steps --eval_steps "$interval")
    default_space="$base_dir/configs/roihu/classifier/hp_space_budget.yaml"
elif [[ -n "${EVALUATIONS:-}" ]]; then
    echo "EVALUATIONS needs TRAIN_DOCUMENTS; epoch-based runs evaluate every epoch" >&2
    exit 1
else
    schedule_args=(--eval_strategy epoch)
    default_space="$base_dir/configs/roihu/classifier/hp_space.yaml"
fi
hp_space="${HP_SPACE:-$default_space}"
dataset_dir="$base_dir/results/datasets/$label_type/$dataset_name"
search_dir="$base_dir/results/models/$label_type/$dataset_name/hp_search/$search_name"
echo "Dataset: $dataset_dir"
echo "Search:  $search_dir (trial $trial, space $hp_space)"
echo "Schedule: ${schedule_args[*]}"

# Fixed settings shared by all trials; the search space sets the rest (learning rate,
# batch size through gradient accumulation, weight decay, warmup, pooling, and in
# hp_space.yaml epochs and label temperature). A searched value replaces the default
# --gradient_accumulation_steps 4 below.
python "$python_script" \
    --model_name jhu-clsp/mmBERT-base \
    --dataset_path "$dataset_dir" \
    --output_dir "$search_dir" \
    --hp_search_space "$hp_space" \
    --hp_trial "$trial" \
    --hp_search_seed 0 \
    --do_train \
    --bf16 \
    --max_length 8192 \
    --decode_url true \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    "${schedule_args[@]}" \
    --save_strategy no \
    --metric_for_best_model js_divergence_bits \
    --greater_is_better false \
    --logging_steps 50 \
    --report_to none \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --seed 42 \
    "$@"
