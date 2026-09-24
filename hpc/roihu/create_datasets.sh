#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=217086
#SBATCH -o ../../logs/eval_%j.out
#SBATCH -e ../../logs/eval_%j.err

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/create_datasets.py"

python $python_script \
    --input $base_dir/results/Qwen3.8-annotations/topic/ \
    --output_dir $base_dir/results/datasets/topic/dataset1