#!/bin/bash
#SBATCH --job-name=create_datasets
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/create_datasets_%j.out
#SBATCH -e ../../logs/create_datasets_%j.err

# Build train/validation/test datasets of soft labels from a label distribution run.
# Usage (submit from this directory so relative log paths resolve):
#   sbatch create_datasets.sh <annotation_run> <topic|format> <dataset_name> [extra create_datasets.py args]
# e.g. sbatch create_datasets.sh Qwen3.8-27B-annotations topic dataset1
#
# Reads results/distribution/<annotation_run>/<label type>/*.jsonl (one file per
# language) and writes results/datasets/<label type>/<dataset_name>. Splits are by
# document ID, so all translations of a document land in the same split.
# The job needs no GPU, but the python-vllm module only runs on the (ARM) GPU nodes.

set -euo pipefail

usage="usage: sbatch create_datasets.sh <annotation_run> <topic|format> <dataset_name> [extra args]"
annotation_run="${1:?$usage}"
label_type="${2:?$usage}"
dataset_name="${3:?$usage}"
shift 3

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/create_datasets.py"
input_dir="$base_dir/results/distribution/$annotation_run/$label_type"
output_dir="$base_dir/results/datasets/$label_type/$dataset_name"
echo "Input:  $input_dir"
echo "Output: $output_dir"

python "$python_script" \
    --input "$input_dir" \
    --label_type "$label_type" \
    --output_dir "$output_dir" \
    "$@"
