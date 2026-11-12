#!/bin/bash
#SBATCH --job-name=create_hplt_dataset
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Build train/validation/test datasets of soft labels from the HPLT label distributions of
# the clean, noisy and poor random samples, with at most 25,000 documents per language from
# clean_random, 25,000 from noisy_random and 10,000 from poor_random (all of them where a
# language has fewer; the job log warns about those files). Roihu version of
# hpc/lumi/create_hplt_dataset.sh, for results copied over from LUMI.
# How to run on Roihu (submit from this directory so relative log paths resolve), once the
# results are in results/distribution/Qwen3.8-27B-hplt-annotations/<collection>_random/:
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch create_hplt_dataset.sh <topic|format> <dataset_name> [extra create_datasets.py args]
# e.g. sbatch create_hplt_dataset.sh topic hplt-random-v1
#      sbatch create_hplt_dataset.sh format hplt-random-v1
#      sbatch create_hplt_dataset.sh topic hplt-random-v1-nourl --no_url
#
# Reads results/distribution/Qwen3.8-27B-hplt-annotations/<collection>_random/<label type>/
# *.jsonl and writes results/datasets/<label type>/<dataset_name> with the columns id,
# language, source (clean_random, noisy_random or poor_random), url, text and labels.
# A language's documents from the three collections are split together, every language in
# exactly 94/1/5 (--stratify_by_language and the ratios below).
# The result files are in the shuffled sample order, so the first N documents of each are
# a random subset of it.
# The job needs no GPU, but the python-vllm module only runs on the (ARM) GPU nodes.

set -euo pipefail

usage="usage: sbatch create_hplt_dataset.sh <topic|format> <dataset_name> [extra create_datasets.py args]"
label_type="${1:?$usage}"
dataset_name="${2:?$usage}"
shift 2
case "$label_type" in
    topic|format) ;;
    *) echo "$usage" >&2; exit 1 ;;
esac

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/create_datasets.py"
annotations="$base_dir/results/distribution/Qwen3.8-27B-hplt-annotations"
output_dir="$base_dir/results/datasets/$label_type/$dataset_name"

# Hugging Face cache directories in the scratch space: building the dataset writes a copy
# of it to the datasets cache, too large for the home directory.
cache_base_dir="/scratch/project_2020507/cache"
export HF_HOME="$cache_base_dir/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# The inputs, their names in the source column, and the documents taken per language
# from each, in the same order.
sources=(clean_random noisy_random poor_random)
inputs=(
    "$annotations/clean_random/$label_type"
    "$annotations/noisy_random/$label_type"
    "$annotations/poor_random/$label_type"
)
max_per_file=(25000 25000 10000)
for input in "${inputs[@]}"; do
    if [[ ! -d "$input" ]]; then
        echo "No results in $input; copy them from LUMI first" >&2
        exit 1
    fi
done
printf 'Input:  %s\n' "${inputs[@]}"
echo "Documents per language: ${max_per_file[*]}"
echo "Output: $output_dir"

python "$python_script" \
    --input "${inputs[@]}" \
    --source_names "${sources[@]}" \
    --max_per_file "${max_per_file[@]}" \
    --label_type "$label_type" \
    --output_dir "$output_dir" \
    --stratify_by_language \
    --train_ratio 0.94 \
    --val_ratio 0.01 \
    --test_ratio 0.05 \
    "$@"
