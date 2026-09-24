#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/eval_%j.out
#SBATCH -e ../../logs/eval_%j.err

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"

label_type="format"

results_dir="$base_dir/results/Qwen3.8-annotations/${label_type}"

python $base_dir/scripts/data_analysis/evaluate_multilingual_labels.py \
  --english $results_dir/english-parallel.jsonl \
  --candidate german=$results_dir/german-parallel.jsonl \
  --candidate finnish=$results_dir/finnish-parallel.jsonl \
  --candidate hungarian=$results_dir/hungarian-parallel.jsonl \
  --candidate icelandic=$results_dir/icelandic-parallel.jsonl \
  --candidate ukrainian=$results_dir/ukrainian-parallel.jsonl \
  --output $base_dir/results/evals/multilingual-label-evaluation-${label_type}.json