#!/bin/bash
#SBATCH --job-name=label
#SBATCH --account=project_462001516
#SBATCH --partition=standard-g
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=120G
#SBATCH --gpus-per-node=8
#SBATCH -o ../../logs/label_%j.out
#SBATCH -e ../../logs/label_%j.err
#SBATCH --array=0-11%6

# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch array_generate_WO_labels.sh
#
# This submits 12 array tasks, one for each of 6 languages and 2 label types:
#   tasks 0-5:  topic labels (english, finnish, german, hungarian, icelandic, ukrainian)
#   tasks 6-11: format labels (the same languages, in the same order)
#   If you change the number of languages or label types, also change the --array argument (num_languages * num_label_types).
# The "%5" limit allows at most 5 tasks to run concurrently. Remove or change it
# only if the GPU allocation and cluster limits allow more concurrent tasks.
#
# Before submitting, set base_dir and python_script below to the paths available
# on the cluster. The output files are written to results/{topic,format}/.

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

python_script="$base_dir/scripts/data_processing/generate_labels.py"
SIF="$base_dir/environments/lumi/vllm.sif"


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
if [[ "$language" == "english" ]]; then
    dataset_split=parallel
elif [[ " ${tower72b_languages[*]} " == *" $language "* ]]; then
    dataset_split=tower72b_parallel
else
    dataset_split=tower9b_parallel
fi

# Models to try
# Qwen/Qwen3.8-27B
# Qwen/Qwen3.5-122B-A10B requires --tensor-parallel-size 8
# google/gemma-4-31B-it
# mistralai/Mistral-Medium-3.5-128B --tensor-parallel-size 8

srun singularity run -B /scratch/project_462001516 "$SIF" python "$python_script" \
    --config "$base_dir/configs/lumi/generate_base.yaml" \
    --model mistralai/Mistral-Medium-3.5-128B \
    --tensor-parallel-size 8 \
    --dataset-config "$dataset_config" \
    --dataset-split "$dataset_split" \
    --language "$language" \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    --output "$base_dir/results/Mistral-Medium-3.5-128B-annotations/${label_type}/${language}-parallel.jsonl"
