#!/bin/bash
#SBATCH --job-name=count_language_tokens
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Count documents and mmBERT tokens per language into a CSV file (columns Language,
# Language code, Documents, Tokens) with scripts/model_training/count_language_tokens.py.
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch count_language_tokens.sh <output name> [count_language_tokens.py args]
#
# With only an output name, it counts the documents create_hplt_dataset.sh takes from the
# HPLT results: 25,000 / 25,000 / 10,000 per language from clean_random, noisy_random and
# poor_random (the topic results; the format results hold the same documents):
#   sbatch count_language_tokens.sh hplt-random
# Other arguments replace these inputs, e.g. a built dataset, or its training split only:
#   sbatch count_language_tokens.sh hplt-random-v1-train --dataset ../../results/datasets/topic/hplt-random-v1 --splits train
# Tokens are the classifier input (decoded URL, blank line, text) without special tokens;
# add --max_length 8192 to count what the model sees in training instead.
#
# Output: results/statistics/<output name>.csv
# The job needs no GPU, but the python-vllm module only runs on the (ARM) GPU nodes.

set -euo pipefail

usage="usage: sbatch count_language_tokens.sh <output name> [count_language_tokens.py args]"
output_name="${1:?$usage}"
shift

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/count_language_tokens.py"
annotations="$base_dir/results/distribution/Qwen3.8-27B-hplt-annotations"
output="$base_dir/results/statistics/${output_name}.csv"

# Hugging Face cache in the scratch space (the mmBERT tokenizer is cached there).
export HF_HOME="/scratch/project_2020507/cache/huggingface"

if (( $# == 0 )); then
    set -- --input "$annotations/clean_random/topic" "$annotations/noisy_random/topic" \
        "$annotations/poor_random/topic" --max_per_file 25000 25000 10000
fi
echo "Arguments: $*"
echo "Output: $output"

# Relative paths in the arguments are relative to this directory (hpc/roihu).
python "$python_script" \
    --languages_file "$base_dir/data/qwen_label_languages.txt" \
    --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 1 )) \
    --output "$output" \
    "$@"
