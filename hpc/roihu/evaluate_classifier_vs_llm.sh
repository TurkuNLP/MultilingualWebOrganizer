#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=project_2020507
#SBATCH --partition=gputest
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/eval_%j.out
#SBATCH -e ../../logs/eval_%j.err

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"

# Models
# Qwen3.8-27B
# Qwen3.5-122B-A10B
# gemma-4-31B-it
# mistral-3.5-128B
# mistralal-3.5-128B-EAGLE 

label_type="format"
model_name="llama3.1-8B"

python $base_dir/scripts/data_analysis/compare_classifier_to_LLM.py \
    --llm_file_dir $base_dir/results/${model_name}-annotations/${label_type}/ \
    --classifier_file $base_dir/results/WO-classifier-annotations/${label_type}/english-parallel.jsonl \
    --output_file $base_dir/results/evals/${model_name}/classifier_llm_${label_type}_comparison_results.json