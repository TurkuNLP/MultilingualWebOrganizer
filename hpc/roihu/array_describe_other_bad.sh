#!/bin/bash
#SBATCH --job-name=describe_other_bad
#SBATCH --account=project_2006385
#SBATCH --partition=gpumedium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/describe_other_bad_%A_%a.out
#SBATCH -e ../../logs/describe_other_bad_%A_%a.err
#SBATCH --array=0-11

# Describe in English, with Qwen/Qwen3.8-27B, the pages whose top topic or format label is
# Other or Bad example, for clustering the descriptions (scripts/data_processing/describe_documents.py).
# One array task per language; the inputs are the train-split pages of all three collections
# (clean, noisy, poor), extracted by scripts/data_analysis/extract_top_label_documents.py.
#
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch array_describe_other_bad.sh
# Smoke test of one language (task 11 is tok_pisin) on 64 pages:
#   sbatch --array=11 --partition=gputest --time=00:15:00 array_describe_other_bad.sh 64
#
# The task number is the language's position (from 0) in the languages list below; the
# list is 4 high-, 4 medium- and 4 low-resource languages from different regions, without
# English. A task that runs out of time or stalls (20 minutes without a finished batch) can
# be submitted again; it resumes from the descriptions already written.
#
# Output: results/other_bad_descriptions/Qwen3.8-27B/<language>.jsonl

module purge
module load python-vllm/0.19.1

set -euo pipefail

max_documents="${1:-}"

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"

languages=(
    # High resource
    russian simplified_chinese arabic spanish
    # Medium resource
    hindi persian thai swahili
    # Low resource (the clean_random sample, capped at 25,000 documents, didn't fill)
    kabuverdianu tibetan waray tok_pisin
)
if (( SLURM_ARRAY_TASK_ID >= ${#languages[@]} )); then
    echo "Array task $SLURM_ARRAY_TASK_ID is out of range for ${#languages[@]} languages" >&2
    exit 1
fi
language=${languages[$SLURM_ARRAY_TASK_ID]}

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Hugginface cache directories in the scratch space.
cache_base_dir="/scratch/project_2020507/cache"
export HF_HOME="$cache_base_dir/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="$cache_base_dir/vllm"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT"

model="Qwen/Qwen3.8-27B"
input_jsonl="$base_dir/results/statistics/other_bad_examples/${language}.jsonl"
# ${model##*/} drops the organization prefix, e.g. Qwen3.8-27B.
output="$base_dir/results/other_bad_descriptions/${model##*/}/${language}.jsonl"
tensor_parallel_size="${SLURM_GPUS_ON_NODE:?no GPU count from SLURM; request GPUs with --gres=gpu:gh200:N}"

max_documents_args=()
if [[ -n "$max_documents" ]]; then
    max_documents_args=(--max-documents "$max_documents")
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "model=$model gpus=$tensor_parallel_size language=$language max_documents=${max_documents:-all}"
echo "input=$input_jsonl ($(wc -l < "$input_jsonl") documents)"
echo "output=$output"

srun python "$base_dir/scripts/data_processing/describe_documents.py" \
    --input-jsonl "$input_jsonl" \
    --output "$output" \
    --model-name "$model" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --max-document-tokens 8192 \
    ${max_documents_args[@]+"${max_documents_args[@]}"}
