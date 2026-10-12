#!/bin/bash
#SBATCH --job-name=label_distributions
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU (288 for 4 GPUs, 144 for 2 GPUs, 72 for 1 GPU)
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/distribution_%j.out
#SBATCH -e ../../logs/distribution_%j.err

# Score one language with label probability distributions.
# Usage (submit from this directory so relative log paths resolve):
#   sbatch generate_distribution.sh <language> <topic|format> <model> [max_documents]
# e.g. sbatch generate_distribution.sh finnish format Qwen/Qwen3.8-27B 2000
#
# Models up to ~31B fit on one GH200 and use the default 1-GPU allocation.
# Larger models need a full node; override the allocation:
#   sbatch --gres=gpu:gh200:4 --cpus-per-task=288 generate_distribution.sh finnish topic mistralai/Mistral-Medium-3.5-128B
# The tensor parallel size always equals the number of allocated GPUs.
#
# Cloudflare/clef (a decision model) is run with generate_label_clef.py instead; it
# uses one GPU, no demonstrations, and a fixed label order:
#   sbatch generate_distribution.sh finnish topic Cloudflare/clef 2000
#
# Models to try
# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen3.8-27B
# google/gemma-4-31B-it
# Qwen/Qwen3.5-122B-A10B (4 GPUs)
# mistralai/Mistral-Medium-3.5-128B (4 GPUs)

module purge
module load python-vllm/0.19.1

set -euo pipefail

usage="usage: sbatch generate_distribution.sh <language> <topic|format> <model> [max_documents]"
lang="${1:?$usage}"
label_type="${2:?$usage}"
model="${3:?$usage}"
max_documents="${4:-}"

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-64}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-${SLURM_CPUS_PER_TASK:-1}}

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Set up Hugging Face cache directories in the scratch space.
export HF_HOME="/scratch/project_2020507/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Define paths to the base directory and Python script.
base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_label_distributions.py"

# Set vLLM cache root to a directory in the scratch space
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$base_dir/cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

# Each language gets its own seed so parallel documents get independent label and
# example orders; otherwise a model's letter preferences push all languages the
# same way and inflate multilingual consistency. English keeps seed 0 so earlier
# English runs (seed 0) remain valid references.
# Same language-to-seed mapping as array_generate_distribution.sh.
languages=(english finnish german hungarian icelandic ukrainian)
seed=""
for index in "${!languages[@]}"; do
    if [[ "${languages[$index]}" == "$lang" ]]; then
        seed=$index
    fi
done
if [[ -z "$seed" ]]; then
    echo "Unknown language: $lang (expected one of: ${languages[*]})" >&2
    exit 1
fi

# Use every allocated GPU for tensor parallelism, so the two cannot disagree.
tensor_parallel_size="${SLURM_GPUS_ON_NODE:?no GPU count from SLURM; request GPUs with --gres=gpu:gh200:N}"

max_documents_args=()
if [[ -n "$max_documents" ]]; then
    max_documents_args=(--max-documents "$max_documents")
fi

# Chat models are scored through vLLM; Clef has its own script. The per-language
# config supplies the dataset; its vLLM model settings are overridden for Clef.
model_args=(--tensor-parallel-size "$tensor_parallel_size" --seed "$seed" --prompt-order rotated)
if [[ "${model,,}" == cloudflare/clef* ]]; then
    python_script="$base_dir/scripts/data_processing/generate_label_clef.py"
    model_args=(--tensor-parallel-size 1 --max-model-len 16384)
    if (( tensor_parallel_size != 1 )); then
        echo "Clef runs on one GPU; got $tensor_parallel_size. Submit with the default 1-GPU allocation." >&2
        exit 1
    fi
fi

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "model=$model gpus=$tensor_parallel_size language=$lang label_type=$label_type seed=$seed max_documents=${max_documents:-config}"
echo "script=${python_script##*/}"

# ${model##*/} drops the organization prefix.
srun python "$python_script" \
            --config "$base_dir/configs/roihu/distribution/generate_${lang}_multisynt.yaml" \
            --model-name "$model" \
            --label-type "$label_type" \
            ${model_args[@]+"${model_args[@]}"} \
            ${max_documents_args[@]+"${max_documents_args[@]}"} \
            --output "$base_dir/results/distribution/${model##*/}-annotations/${label_type}/${lang}-parallel.jsonl"
