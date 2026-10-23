#!/bin/bash
#SBATCH --job-name=distribution_hplt
#SBATCH --account=project_462001516
#SBATCH --partition=small-g
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14 # 7 per GPU
#SBATCH --mem=60G
#SBATCH --gpus-per-node=2
#SBATCH -o ../../logs/%x_%A_%a.out
#SBATCH -e ../../logs/%x_%A_%a.err
#SBATCH --array=0-255%36

# Label the HPLT samples of data/hplt_<data_quality>_sample (made by sample_hplt_by_quality.sh,
# or collected from hplt-sampler runs for the *_random samples). data_quality is clean_random
# unless DATA_QUALITY is set when submitting: DATA_QUALITY=noisy_random sbatch ...
# For the random samples, run reuse_hplt_labels.sh first to reuse earlier labels, and label
# the whole sample so that it stays a random sample.
# with topic and format label distributions from Qwen/Qwen3.8-27B.
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch array_generate_distribution_hplt.sh [max_documents]
#
# The languages come from the file set in languages_file below, by default the 128 of
# data/qwen_label_languages.txt. This array has 250 tasks, one per language and label type,
# each on 2 GPUs (line N of the file counts from 0, blank lines excluded):
#   tasks 0-127:   topic  (language N of the file is task N)
#   tasks 128-255: format (language N of the file is task 128 + N)
# Run a subset with --array, e.g. English and Finnish, both label types:
#   sbatch --array=29,35,157,163 array_generate_distribution_hplt.sh
# With another languages file, also change the --array argument (2 * the number of languages).
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

set -euo pipefail

max_documents="${1:-}"

module purge
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Hugginface cache directories in the scratch space.
cache_base_dir="/scratch/project_462001516/cache"
export HF_HOME="$cache_base_dir/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="$cache_base_dir/vllm"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT"

python_script="$base_dir/scripts/data_processing/generate_label_distributions.py"
config="$base_dir/configs/lumi/distribution/generate_hplt.yaml"
SIF="$base_dir/environments/lumi/vllm.sif"
model="Qwen/Qwen3.8-27B"

# Language names: the part of each non-empty line before the colon, in file order.
# data/oellm_languages.txt has the 38 languages of the earlier clean, noisy and poor samples.
languages_file="$base_dir/data/qwen_label_languages.txt"
mapfile -t language_names < <(sed -n 's/^[[:space:]]*\([^:]*[^:[:space:]]\)[[:space:]]*:.*/\1/p' "$languages_file")
num_languages=${#language_names[@]}

language_index=$SLURM_ARRAY_TASK_ID
label_type=topic

if (( SLURM_ARRAY_TASK_ID >= num_languages )); then
    language_index=$((SLURM_ARRAY_TASK_ID - num_languages))
    label_type=format
fi
if (( language_index >= num_languages )); then
    echo "Array task $SLURM_ARRAY_TASK_ID is out of range for $num_languages languages and 2 label types" >&2
    exit 1
fi

# clean, noisy or poor; clean_random, noisy_random or poor_random for the random samples.
# Set DATA_QUALITY when submitting to choose another, e.g. DATA_QUALITY=noisy_random sbatch ...
data_quality="${DATA_QUALITY:-clean_random}"
language_name=${language_names[$language_index]}
# The sample file name: lower case, spaces as underscores (as the samplers name them).
language=${language_name,,}
language=${language// /_}
input_jsonl="$base_dir/data/hplt_${data_quality}_sample/${language}.jsonl"
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
echo "input=$input_jsonl"
echo "output=$output"

if [[ ! -f "$input_jsonl" ]]; then
    echo "No sample file for $language: $input_jsonl" >&2
    exit 1
fi

srun singularity run -B /scratch/project_462001516 "$SIF" python "$python_script" \
    --config "$config" \
    --model-name "$model" \
    --tensor-parallel-size "$tensor_parallel_size" \
    --input-jsonl "$input_jsonl" \
    --language "$language_name" \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    --prompt-order rotated \
    ${max_documents_args[@]+"${max_documents_args[@]}"} \
    --output "$output"
