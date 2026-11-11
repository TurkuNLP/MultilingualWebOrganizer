#!/bin/bash
#SBATCH --job-name=labelling_status
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Report which HPLT random-sample labelling results are complete (25,000 documents for
# clean and noisy, 10,000 for poor, or the whole sample if it is smaller), which are
# still running, and which need a relaunch, with the array task IDs and submit commands.
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch labelling_status.sh [extra labelling_status.py args]
# Writes results/distribution/Qwen3.8-27B-hplt-annotations/labelling_status.{tsv,md}.
# It reads every result file (about 80 GB), so it runs as a job, not on a login node.

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
SIF="$base_dir/environments/lumi/vllm.sif"
workers="$SLURM_CPUS_PER_TASK"

srun singularity run -B /scratch/project_462001516 "$SIF" python \
    "$base_dir/scripts/data_processing/labelling_status.py" --base-dir "$base_dir" --workers "$workers" "$@"
