#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/eval_%j.out
#SBATCH -e ../../logs/eval_%j.err

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"

label_type="format"

python $base_dir/scripts/data_analysis/compare_classifier_to_LLM.py \
    --llm_file_dir $base_dir/results/Qwen3.8-annotations/${label_type}/ \
    --classifier_file $base_dir/results/WO-classifier-annotations/${label_type}/english-parallel.jsonl \
    --output_file $base_dir/results/evals/classifier_llm_${label_type}_comparison_results.json