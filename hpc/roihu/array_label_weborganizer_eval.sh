#!/bin/bash
#SBATCH --job-name=label_weborganizer_eval
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU (288 for 4 GPUs, 144 for 2 GPUs, 72 for 1 GPU)
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%A_%a.out
#SBATCH -e ../../logs/%x_%A_%a.err
#SBATCH --array=0-1

# Label WebOrganizer's evaluation pages (10K English pages per split) with Qwen3.8-27B, using
# exactly the settings of our classifier training labels (array_generate_distribution_hplt.sh):
# configs/roihu/distribution/generate_hplt.yaml (8,192-token truncation, max_model_len 32768,
# seed 0), rotated prompt order, both example sets, and the page URL in the prompt. Unlike
# array_generate_distribution_weborganizer.sh (model screening), it does not truncate to
# 50,000 characters the way WebOrganizer's annotator did.
#
# Tasks: 0 = topic (TopicAnnotations-Llama-3.1-405B-FP8), 1 = format (FormatAnnotations-...).
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch array_label_weborganizer_eval.sh                 # validation split (Table 7)
#   SPLIT=test sbatch array_label_weborganizer_eval.sh      # test split
# A smoke test: sbatch --partition=gputest --time=00:15:00 array_label_weborganizer_eval.sh 200
# (then submit the full run without the number; it resumes from the records already written).
# A task that runs out of time or stalls can be submitted again; it resumes.
#
# Output: results/distribution/Qwen3.8-27B-weborganizer-annotations/{topic,format}/<split>.jsonl
# Then build evaluation datasets with these labels as targets:
#   TARGETS=qwen sbatch create_weborganizer_eval_dataset.sh

module purge
module load python-vllm/0.19.1

set -euo pipefail

max_documents="${1:-}"
split="${SPLIT:-validation}"

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_label_distributions.py"
config="$base_dir/configs/roihu/distribution/generate_hplt.yaml"
model="Qwen/Qwen3.8-27B"

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

label_types=(topic format)
datasets=(WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8 WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8)
if (( SLURM_ARRAY_TASK_ID >= ${#label_types[@]} )); then
    echo "Array task $SLURM_ARRAY_TASK_ID is out of range; use --array=0-1" >&2
    exit 1
fi
label_type=${label_types[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
# ${model##*/} drops the organization prefix, e.g. Qwen3.8-27B.
output="$base_dir/results/distribution/${model##*/}-weborganizer-annotations/${label_type}/${split}.jsonl"

# Use every allocated GPU for tensor parallelism, so the two cannot disagree.
tensor_parallel_size="${SLURM_GPUS_ON_NODE:?no GPU count from SLURM; request GPUs with --gres=gpu:gh200:N}"

max_documents_args=(--max-documents 10000)
if [[ -n "$max_documents" ]]; then
    max_documents_args=(--max-documents "$max_documents")
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "model=$model gpus=$tensor_parallel_size dataset=$dataset split=$split label_type=$label_type max_documents=${max_documents:-all}"
echo "output=$output"

srun python "$python_script" \
    --config "$config" \
    --model-name "$model" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --input-dataset "$dataset" \
    --dataset-split "$split" \
    --language English \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    --prompt-order rotated \
    "${max_documents_args[@]}" \
    --output "$output"

echo "Done at $(date)"
