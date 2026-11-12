#!/bin/bash
#SBATCH --job-name=create_hplt_dataset
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Build train/validation/test datasets of soft labels from the HPLT label distributions of
# the clean, noisy and poor random samples, with at most 25,000 documents per language from
# clean_random, 25,000 from noisy_random and 10,000 from poor_random (all of them where a
# language has fewer; the job log warns about those files).
# How to run on LUMI (submit from this directory so relative log paths resolve), once the
# labelling runs have finished:
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
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
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/create_datasets.py"
SIF="$base_dir/environments/lumi/vllm.sif"
annotations="$base_dir/results/distribution/Qwen3.8-27B-hplt-annotations"
output_dir="$base_dir/results/datasets/$label_type/$dataset_name"

# Hugginface cache directories in the scratch space, not in the home directory.
cache_base_dir="/scratch/project_462001516/cache"
export HF_HOME="$cache_base_dir/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# The inputs, their names in the source column, and the documents taken per language
# from each, in the same order.
sources=(clean_random noisy_random poor_random)
inputs=(
    "$annotations/clean_random/$label_type"
    "$annotations/noisy_random/$label_type"
    "$annotations/poor_random/$label_type"
)
max_per_file=(25000 25000 10000)
printf 'Input:  %s\n' "${inputs[@]}"
echo "Documents per language: ${max_per_file[*]}"
echo "Output: $output_dir"

srun singularity run -B /scratch/project_462001516 "$SIF" python "$python_script" \
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
