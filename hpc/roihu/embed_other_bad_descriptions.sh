#!/bin/bash
#SBATCH --job-name=embed_other_bad
#SBATCH --account=project_2006385
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/embed_other_bad_%j.out
#SBATCH -e ../../logs/embed_other_bad_%j.err

# Embed the English descriptions made by array_describe_other_bad.sh with
# Qwen/Qwen3-Embedding-8B (scripts/data_processing/embed_descriptions.py): one array per
# description field (genre, topic, problems, description), 1024 dimensions each.
# Run it after every description task has finished: the script stops if a language's
# descriptions don't cover all of its extracted pages.
#
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch embed_other_bad_descriptions.sh
# Or chain it to the description array:
#   sbatch --dependency=afterok:<array job id> embed_other_bad_descriptions.sh
#
# Output: results/other_bad_descriptions/Qwen3.8-27B/embeddings/Qwen3-Embedding-8B/
#   {index.tsv, metadata.json, genre.npy, topic.npy, problems.npy, description.npy}

module purge
module load python-vllm/0.19.1

set -euo pipefail

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"

# The same languages as array_describe_other_bad.sh.
languages=(
    russian simplified_chinese arabic spanish
    hindi persian thai swahili
    kabuverdianu tibetan waray tok_pisin
)

export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Hugginface cache directories in the scratch space.
cache_base_dir="/scratch/project_2020507/cache"
export HF_HOME="$cache_base_dir/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="$cache_base_dir/vllm"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT"

model="Qwen/Qwen3-Embedding-8B"
descriptions_dir="$base_dir/results/other_bad_descriptions/Qwen3.8-27B"
output_dir="$descriptions_dir/embeddings/${model##*/}"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "model=$model languages=${languages[*]}"
echo "output=$output_dir"

srun python "$base_dir/scripts/data_processing/embed_descriptions.py" \
    --descriptions-dir "$descriptions_dir" \
    --input-dir "$base_dir/results/statistics/other_bad_examples" \
    --languages "${languages[@]}" \
    --output-dir "$output_dir" \
    --model-name "$model" \
    --dimensions 1024
