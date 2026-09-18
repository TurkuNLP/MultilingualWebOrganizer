#!/bin/bash
#SBATCH --job-name=discover_topic_labels
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=144
#SBATCH --gres=gpu:gh200:2
#SBATCH --mem=434172
#SBATCH -o ../../logs/label_%j.out
#SBATCH -e ../../logs/label_%j.err

module purge
module load python-vllm/0.19.1

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-64}
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-${SLURM_CPUS_PER_TASK:-1}}

# Keep expected third-party startup warnings out of the SLURM error log.
export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

# Set up Hugging Face cache directories in the scratch space.
export HF_HOME="/scratch/project_2020507/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Define paths to the base directory and Python script.
base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_labels.py"

# Set vLLM cache root to a directory in the scratch space
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$base_dir/cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

srun python "$python_script" \
            --config "$base_dir/configs/roihu/generate_nemotron-cc.yaml"
