#!/bin/bash
#SBATCH --job-name=hplt_distribution
#SBATCH --account=project_2006385
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU (288 for 4 GPUs, 144 for 2 GPUs, 72 for 1 GPU)
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/hplt_%A_%a.out
#SBATCH -e ../../logs/hplt_%A_%a.err
#SBATCH --array=0-99%50

# Label the HPLT samples of data/hplt_<data_quality>_sample (made by sample_hplt_by_quality.sh)
# with topic and format label distributions from Qwen/Qwen3.8-27B.
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch array_generate_distribution_hplt.sh [max_documents]
#
# The languages are the sample files (<language>.jsonl) of the chosen sample, sorted by
# name; set data_quality below to choose the sample. The poor sample has 125 languages
# of 10,000 documents each, so the array has 250 tasks, each on 1 GPU:
#   tasks 0-124:   topic  (the N-th language, counting from 0, is task N)
#   tasks 125-249: format (the N-th language is task 125 + N)
# Find a language's number (counting from 0) with:
#   ls ../../data/hplt_poor_sample/*.jsonl | LC_ALL=C sort | grep -n -e /english. -e /finnish. | awk -F: '{print $1 - 1, $2}'
# Run a subset with --array, e.g. English (27) and Finnish (32), both label types:
#   sbatch --array=27,32,152,157 array_generate_distribution_hplt.sh
# For a sample with another number of languages N, override the array on the command
# line: sbatch --array=0-<2N-1>%36 array_generate_distribution_hplt.sh
# Don't add or remove sample files while an array is running: tasks are matched to
# languages by their position in the sorted file list.
#
# max_documents defaults to the config value (run.max_documents in generate_hplt.yaml).
# The samples are in shuffled order, so a smaller number labels a mixed subset of each.
# A task that runs out of time can be submitted again; it resumes from the results it
# has already written.
# A task that makes no progress for 20 minutes (run.stall_timeout_minutes) exits with
# code 1 and a traceback in its .err log showing where it hung; resubmit it to resume.
# A task whose language has no sample file stops with an error.
#
# Prompts use --prompt-order rotated so vLLM can reuse the shared prompt prefix.
#
# Output: results/distribution/Qwen3.8-27B-hplt-annotations/<data_quality>/{topic,format}/<language>.jsonl

module purge
module load python-vllm/0.19.1

set -euo pipefail

max_documents="${1:-}"

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"

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

python_script="$base_dir/scripts/data_processing/generate_label_distributions.py"
config="$base_dir/configs/roihu/distribution/generate_hplt.yaml"
model="Qwen/Qwen3.8-27B"

data_quality=poor # clean, noisy or poor
sample_dir="$base_dir/data/hplt_${data_quality}_sample"

# The languages are the sample files, sorted byte-wise (LC_ALL=C) so the order is the
# same on every node; names such as norwegian_bokmål and māori are UTF-8.
mapfile -t languages < <(
    find "$sample_dir" -maxdepth 1 -name '*.jsonl' -printf '%f\n' | sed 's/\.jsonl$//' | LC_ALL=C sort
)
if (( ${#languages[@]} == 0 )); then
    echo "No sample files (*.jsonl) in $sample_dir" >&2
    exit 1
fi

language_index=$SLURM_ARRAY_TASK_ID
label_type=topic

if (( SLURM_ARRAY_TASK_ID >= ${#languages[@]} )); then
    language_index=$((SLURM_ARRAY_TASK_ID - ${#languages[@]}))
    label_type=format
fi
if (( language_index >= ${#languages[@]} )); then
    echo "Array task $SLURM_ARRAY_TASK_ID is out of range for ${#languages[@]} languages and 2 label types" >&2
    exit 1
fi

language=${languages[$language_index]}
input_jsonl="$sample_dir/${language}.jsonl"
# ${model##*/} drops the organization prefix, e.g. Qwen3.8-27B.
output="$base_dir/results/distribution/${model##*/}-hplt-annotations/${data_quality}/${label_type}/${language}.jsonl"
# Use every allocated GPU for tensor parallelism, so the two cannot disagree.
tensor_parallel_size="${SLURM_GPUS_ON_NODE:?no GPU count from SLURM; request GPUs with --gpus-per-node=N}"

max_documents_args=()
if [[ -n "$max_documents" ]]; then
    max_documents_args=(--max-documents "$max_documents")
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "model=$model gpus=$tensor_parallel_size language=$language label_type=$label_type max_documents=${max_documents:-config}"
echo "sample=$sample_dir (${#languages[@]} languages; this is number $language_index)"
echo "input=$input_jsonl"
echo "output=$output"

if [[ ! -f "$input_jsonl" ]]; then
    echo "No sample file for $language: $input_jsonl" >&2
    exit 1
fi

srun python "$python_script" \
    --config "$config" \
    --model-name "$model" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --input-jsonl "$input_jsonl" \
    --language "${language//_/ }" \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    --prompt-order rotated \
    ${max_documents_args[@]+"${max_documents_args[@]}"} \
    --output "$output"
