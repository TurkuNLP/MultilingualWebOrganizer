#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=288 # 72 CPUs per GPU (288 for 4 GPUs, 144 for 2 GPUs, 72 for 1 GPU)
#SBATCH --gres=gpu:gh200:4
#SBATCH --mem=434172
#SBATCH -o ../../logs/label_%j.out
#SBATCH -e ../../logs/label_%j.err
#SBATCH --array=0-11%6

# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
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
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_labels.py"

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
# Qwen/Qwen3.5-122B-A10B requires --tensor-parallel-size 4
# google/gemma-4-31B-it
# mistralai/Mistral-Medium-3.5-128B --tensor-parallel-size 4

# Clear cache after running the larger models.

srun python "$python_script" \
    --config "$base_dir/configs/roihu/generate_base.yaml" \
    --model Qwen/Qwen3.5-122B-A10B \
    --tensor-parallel-size 4 \
    --dataset-config "$dataset_config" \
    --dataset-split "$dataset_split" \
    --language "$language" \
    --label-type "$label_type" \
    --labels "$base_dir/scripts/data_processing/${label_type}s.yaml" \
    --examples "$base_dir/scripts/data_processing/${label_type}_examples.yaml" \
    --output "$base_dir/results/Qwen3.5-122B-A10B-annotations/${label_type}/${language}-parallel.jsonl"
