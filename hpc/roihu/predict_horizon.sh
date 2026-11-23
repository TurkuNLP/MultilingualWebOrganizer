#!/bin/bash
#SBATCH --job-name=predict_horizon
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Predict with Horizon Labs' multilingual WebOrganizer classifier
# (Horizon-Labs/multilingual-web-topic-format-base: mmBERT-base, text only, 512 tokens, 48 outputs)
# on a split of one of our datasets, with scripts/model_evaluation/predict_horizon.py. The
# dataset's label type selects the model's topic_* or format_* outputs.
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch predict_horizon.sh <topic|format> <dataset name> <split> [predict_horizon.py args]
# e.g. their own English evaluation (3,000 WebOrganizer test pages, Llama labels; should give
# their published accuracy, topic 0.8173 and format 0.688):
#   RUN_NAME=horizon-base-test-sample3000 sbatch --partition=gputest --time=00:15:00 \
#       predict_horizon.sh topic weborganizer-405B test --sample_documents 3000 --max_characters 6000
# WebOrganizer's validation split (10K pages) and our test split (364K pages, 128 languages):
#   sbatch --partition=gputest --time=00:15:00 predict_horizon.sh topic weborganizer-405B validation
#   sbatch predict_horizon.sh topic hplt-finetune test
#
# Output: results/models/<label type>/<dataset name>/${RUN_NAME:-horizon-base-<split>}/
# (predictions.jsonl in finetune.py's format, metrics.json on the shared labels; the log ends
# with the accuracy and their macro-F1).

set -euo pipefail

usage="usage: sbatch predict_horizon.sh <topic|format> <dataset name> <split> [predict_horizon.py args]"
label_type="${1:?$usage}"
dataset_name="${2:?$usage}"
split="${3:?$usage}"
shift 3
case "$label_type" in
    topic|format) ;;
    *) echo "$usage" >&2; exit 1 ;;
esac
run_name="${RUN_NAME:-horizon-base-$split}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_evaluation/predict_horizon.py"
dataset_dir="$base_dir/results/datasets/$label_type/$dataset_name"
if [[ ! -f "$dataset_dir/labels.json" ]]; then
    echo "No dataset at $dataset_dir" >&2
    exit 1
fi
if ! grep -q "\"label_type\": \"$label_type\"" "$dataset_dir/labels.json"; then
    echo "$dataset_dir/labels.json is not a $label_type dataset" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

output_dir="$base_dir/results/models/$label_type/$dataset_name/$run_name"
echo "== Horizon-Labs/multilingual-web-topic-format-base on $dataset_dir ($split) -> $output_dir"
python "$python_script" \
    --dataset_path "$dataset_dir" \
    --split "$split" \
    --output_dir "$output_dir" \
    "$@"
echo "Done at $(date)"
