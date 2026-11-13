#!/bin/bash
#SBATCH --job-name=predict_weborganizer
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Predict with WebOrganizer's original classifiers (WebOrganizer/<Topic|Format>Classifier with
# URL and text, and -NoURL with text only) on a split of one of our datasets, with
# scripts/model_evaluation/predict_weborganizer.py. Runs both input variants, one after the
# other (VARIANTS="url nourl" by default).
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch predict_weborganizer.sh <topic|format> <dataset name> <split> [predict_weborganizer.py args]
# e.g. their models on their own validation split (reproduces their Table 7; 10K documents):
#   sbatch --partition=gputest --time=00:15:00 predict_weborganizer.sh topic weborganizer-405B validation
# and on our test split (364K documents in 128 languages):
#   sbatch predict_weborganizer.sh topic hplt-finetune test
# A quick test on 500 documents:
#   VARIANTS=url sbatch --partition=gputest --time=00:15:00 predict_weborganizer.sh topic hplt-finetune test \
#       --max_documents 500 --overwrite
# The dataset needs a url column for the URL model (the -nourl datasets have none; the text-only
# model ignores it, so use the dataset with URLs for both).
#
# Output: results/models/<label type>/<dataset name>/weborganizer-<url|nourl>-<split>/
# (predictions.jsonl in finetune.py's format, metrics.json on the shared labels; the log ends
# with the accuracy on documents whose target top label is > 0.75).
# Time: check the documents-per-second lines of the first run before submitting the large one.

set -euo pipefail

usage="usage: sbatch predict_weborganizer.sh <topic|format> <dataset name> <split> [predict_weborganizer.py args]"
label_type="${1:?$usage}"
dataset_name="${2:?$usage}"
split="${3:?$usage}"
shift 3
case "$label_type" in
    topic) model_base="WebOrganizer/TopicClassifier" ;;
    format) model_base="WebOrganizer/FormatClassifier" ;;
    *) echo "$usage" >&2; exit 1 ;;
esac
variants="${VARIANTS:-url nourl}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_evaluation/predict_weborganizer.py"
dataset_dir="$base_dir/results/datasets/$label_type/$dataset_name"
if [[ ! -f "$dataset_dir/labels.json" ]]; then
    echo "No dataset at $dataset_dir" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

for variant in $variants; do
    case "$variant" in
        url) model="$model_base" ;;
        nourl) model="$model_base-NoURL" ;;
        *) echo "VARIANTS must hold url and/or nourl, got '$variant'" >&2; exit 1 ;;
    esac
    output_dir="$base_dir/results/models/$label_type/$dataset_name/weborganizer-$variant-$split"
    echo "== $model on $dataset_dir ($split) -> $output_dir"
    python "$python_script" \
        --model "$model" \
        --dataset_path "$dataset_dir" \
        --split "$split" \
        --output_dir "$output_dir" \
        "$@"
done
echo "Done at $(date)"
