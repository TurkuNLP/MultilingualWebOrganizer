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

label_type="topic"

# Models
# Qwen3.8-27B
# Qwen3.5-122B-A10B
# gemma-4-31B-it
# mistral-3.5-128B
# mistralal-3.5-128B-EAGLE
model_name="Qwen3.8-27B"

results_dir="$base_dir/results/${model_name}-annotations/${label_type}"

python $base_dir/scripts/data_analysis/evaluate_multilingual_labels.py \
  --english $results_dir/english-parallel.jsonl \
  --candidate german=$results_dir/german-parallel.jsonl \
  --candidate finnish=$results_dir/finnish-parallel.jsonl \
  --candidate hungarian=$results_dir/hungarian-parallel.jsonl \
  --candidate icelandic=$results_dir/icelandic-parallel.jsonl \
  --candidate ukrainian=$results_dir/ukrainian-parallel.jsonl \
  --output $base_dir/results/evals/${model_name}/multilingual-label-evaluation-${label_type}.json