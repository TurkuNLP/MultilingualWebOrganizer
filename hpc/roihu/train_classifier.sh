#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=217086
#SBATCH -o ../../logs/train_%j.out
#SBATCH -e ../../logs/train_%j.err

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/finetune.py"

python $python_script \
    --model_name "jhu-clsp/mmBERT-base" \
    --do-train \
    --num_labels 24 \
    --dataset_path $base_dir/results/datasets/topic/dataset1 \
    --output_dir $base_dir/results/models/topic/dataset1
