#!/bin/bash
#SBATCH --job-name=label_distributions
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU (288 for 4 GPUs, 144 for 2 GPUs, 72 for 1 GPU)
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/distribution_%A_%a.out
#SBATCH -e ../../logs/distribution_%A_%a.err
#SBATCH --array=0-11%6

# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch array_generate_distribution.sh <model> [max_documents]
#
# Models up to ~31B (e.g. meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3.8-27B,
# google/gemma-4-31B-it) fit on one GH200 (96 GB) and use the default 1-GPU
# allocation. Larger models need a full node; override the allocation:
# Qwen/Qwen3.5-122B-A10B 
#   sbatch --gres=gpu:gh200:4 --cpus-per-task=288 array_generate_distribution.sh mistralai/Mistral-Medium-3.5-128B
# The tensor parallel size always equals the number of allocated GPUs.
#
# This array has 12 tasks, one for each of 6 languages and 2 label types:
#   tasks 0-5:  topic  (0 english, 1 finnish, 2 german, 3 hungarian, 4 icelandic, 5 ukrainian)
#   tasks 6-11: format (6 english, 7 finnish, 8 german, 9 hungarian, 10 icelandic, 11 ukrainian)
# Run a subset with --array, e.g. the screening stage (english, finnish, ukrainian) on 2000 documents:
#   sbatch --array=0,1,5,6,7,11 array_generate_distribution.sh Qwen/Qwen3.8-27B 2000
# max_documents defaults to the config value (10000).
#
# Prompts use --prompt-order rotated so vLLM can reuse the shared prompt prefix.
# Existing outputs made with the default random order cannot be resumed with it
# (the script stops with an error); leave those tasks out of --array, e.g. task 0
# for models whose English topic results already exist.
# The "%6" limit allows at most 6 tasks to run concurrently.
#
# Cloudflare/clef is a decision model, not a chat model: it is run with
# generate_label_clef.py and configs/roihu/distribution/generate_clef.yaml instead.
# It fits on one GH200, takes no demonstrations, and always presents the labels in
# the same (alphabetical) order, so --prompt-order and the per-language seed don't apply:
#   sbatch array_generate_distribution.sh Cloudflare/clef 2000
#
# Output: results/distribution/<model>-annotations/{topic,format}/<language>-parallel.jsonl

module purge
module load python-vllm/0.19.1

set -euo pipefail

model="${1:?usage: sbatch array_generate_distribution.sh <model> [max_documents]}"
max_documents="${2:-}"

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_label_distributions.py"

languages=(english finnish german hungarian icelandic ukrainian)
dataset_configs=(eng_Latn fin_Latn deu_Latn hun_Latn isl_Latn ukr_Cyrl)

# These languages have been translated with the tower72b model
tower72b_languages=(finnish german)

language_index=$SLURM_ARRAY_TASK_ID
label_type=topic

if (( SLURM_ARRAY_TASK_ID >= ${#languages[@]} )); then
    language_index=$((SLURM_ARRAY_TASK_ID - ${#languages[@]}))
    label_type=format
fi

language=${languages[$language_index]}
dataset_config=${dataset_configs[$language_index]}
# Each language gets its own seed so parallel documents get independent label and
# example orders; otherwise a model's letter preferences push all languages the
# same way and inflate multilingual consistency. English keeps seed 0 so earlier
# English runs (seed 0) remain valid references.
seed=$language_index
if [[ "$language" == "english" ]]; then
    dataset_split=parallel
elif [[ " ${tower72b_languages[*]} " == *" $language "* ]]; then
    dataset_split=tower72b_parallel
else
    dataset_split=tower9b_parallel
fi

# Use every allocated GPU for tensor parallelism, so the two cannot disagree.
tensor_parallel_size="${SLURM_GPUS_ON_NODE:?no GPU count from SLURM; request GPUs with --gres=gpu:gh200:N}"

max_documents_args=()
if [[ -n "$max_documents" ]]; then
    max_documents_args=(--max-documents "$max_documents")
fi

# Chat models are scored through vLLM; Clef has its own script and config.
config="$base_dir/configs/roihu/distribution/generate_base.yaml"
model_args=(--tensor-parallel-size "$tensor_parallel_size" --seed "$seed" --prompt-order rotated)
if [[ "${model,,}" == cloudflare/clef* ]]; then
    python_script="$base_dir/scripts/data_processing/generate_label_clef.py"
    config="$base_dir/configs/roihu/distribution/generate_clef.yaml"
    model_args=()
    if (( tensor_parallel_size != 1 )); then
        echo "Clef runs on one GPU; got $tensor_parallel_size. Submit with the default 1-GPU allocation." >&2
        exit 1
    fi
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "model=$model gpus=$tensor_parallel_size language=$language label_type=$label_type seed=$seed max_documents=${max_documents:-config}"
echo "script=${python_script##*/} config=${config##*/}"

# The distribution script checks at startup that the answer letters are single
# tokens for the chosen model and stops with an error if they are not.
# ${model##*/} drops the organization prefix, e.g. Qwen3.8-27B.
srun python "$python_script" \
    --config "$config" \
    --model-name "$model" \
    --dataset-config "$dataset_config" \
    --dataset-split "$dataset_split" \
    --language "$language" \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    ${model_args[@]+"${model_args[@]}"} \
    ${max_documents_args[@]+"${max_documents_args[@]}"} \
    --output "$base_dir/results/distribution/${model##*/}-annotations/${label_type}/${language}-parallel.jsonl"
