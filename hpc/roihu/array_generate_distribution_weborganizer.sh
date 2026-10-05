#!/bin/bash
#SBATCH --job-name=label_distributions_wo
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU (288 for 4 GPUs, 144 for 2 GPUs, 72 for 1 GPU)
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/distribution_wo_%A_%a.out
#SBATCH -e ../../logs/distribution_wo_%A_%a.err
#SBATCH --array=0-1

# Label the documents of WebOrganizer's Llama-3.1-405B annotations, to compare a
# model's label distributions with the original WebOrganizer labels.
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch array_generate_distribution_weborganizer.sh <model> [max_documents]
# Larger models need a full node:
#   sbatch --gres=gpu:gh200:4 --cpus-per-task=288 array_generate_distribution_weborganizer.sh mistralai/Mistral-Medium-3.5-128B
#
# Tasks: 0 = topic (TopicAnnotations-Llama-3.1-405B-FP8), 1 = format (FormatAnnotations-Llama-3.1-405B-FP8).
# max_documents defaults to 2000, the screening size. Documents are truncated to
# 50,000 characters with WebOrganizer's truncation marker, so the model sees the
# same text as WebOrganizer's annotator. Document IDs are the WARC record IDs.
#
# Output: results/distribution/weborganizer-405B/<model>/{topic,format}.jsonl

module purge
module load python-vllm/0.19.1

set -euo pipefail

model="${1:?usage: sbatch array_generate_distribution_weborganizer.sh <model> [max_documents]}"
max_documents="${2:-2000}"

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_label_distributions.py"

label_types=(topic format)
datasets=(WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8 WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8)
label_type=${label_types[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

# Use every allocated GPU for tensor parallelism, so the two cannot disagree.
tensor_parallel_size="${SLURM_GPUS_ON_NODE:?no GPU count from SLURM; request GPUs with --gres=gpu:gh200:N}"

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "model=$model gpus=$tensor_parallel_size dataset=$dataset label_type=$label_type max_documents=$max_documents"

# ${model##*/} drops the organization prefix, e.g. Qwen3.8-27B.
srun python "$python_script" \
    --config "$base_dir/configs/roihu/distribution/generate_base.yaml" \
    --model "$model" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --input-dataset "$dataset" \
    --dataset-split train \
    --language English \
    --seed 0 \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    --max-document-chars 50000 \
    --prompt-order rotated \
    --max-documents "$max_documents" \
    --output "$base_dir/results/distribution/weborganizer-405B/${model##*/}/${label_type}.jsonl"
