#!/bin/bash
#SBATCH --job-name=hp_search_classifier
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/hp_search_%A_%a.out
#SBATCH -e ../../logs/hp_search_%A_%a.err

# One trial of a random hyperparameter search per array task (trial = array task ID).
# Usage (submit from this directory so relative log paths resolve):
#   sbatch --array=0-15%4 hp_search_classifier.sh <topic|format> <dataset_name> <search_name> [extra finetune.py args]
# e.g. sbatch --array=0-15%4 hp_search_classifier.sh topic dataset1 search1 --max_train_samples 20000
#
# %4 runs at most four trials at a time. Each trial trains on the training split (or a
# fixed random subset with --max_train_samples), evaluates on the validation split
# every epoch, and is scored by its best js_divergence_bits. Trials save no
# checkpoints and never touch the test split.
#
# Search space: configs/roihu/classifier/hp_space.yaml (override with HP_SPACE=<path>).
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
hp_space="${HP_SPACE:-$base_dir/configs/roihu/classifier/hp_space.yaml}"
dataset_dir="$base_dir/results/datasets/$label_type/$dataset_name"
search_dir="$base_dir/results/models/$label_type/$dataset_name/hp_search/$search_name"
echo "Dataset: $dataset_dir"
echo "Search:  $search_dir (trial $trial, space $hp_space)"

# Fixed settings shared by all trials; the search space sets the rest (learning rate,
# weight decay, warmup, epochs, gradient accumulation, label temperature, pooling).
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --eval_strategy epoch \
    --save_strategy no \
    --metric_for_best_model js_divergence_bits \
    --greater_is_better false \
    --logging_steps 50 \
    --report_to none \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --seed 42 \
    "$@"
