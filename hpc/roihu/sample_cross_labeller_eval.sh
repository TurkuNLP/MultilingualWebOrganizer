#!/bin/bash
#SBATCH --job-name=sample_cross_labeller_eval
#SBATCH --account=project_2020507
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Draw the pages that other labellers (Horizon Labs' teacher prompt, a neutral judge) label for
# the cross-labeller comparison, with scripts/data_processing/sample_cross_labeller_eval.py:
# --per_language pages per language of a split of hplt-finetune, duplicates of training pages
# left out, seeded.
# How to run on Roihu (submit from a Roihu-CPU login node, from this directory):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch sample_cross_labeller_eval.sh [per_language] [sample_cross_labeller_eval.py args]
# e.g. 150 pages per language of the test split (19,200 pages; about 2 minutes):
#   sbatch sample_cross_labeller_eval.sh 150
# 1,000 more pages in each of 10 major languages, none of them already in hplt-test-150
# (SAMPLE_NAME names the output directory; paths in the extra arguments are relative to the
# repository root, where the job runs):
#   SAMPLE_NAME=hplt-test-major10-1000 sbatch sample_cross_labeller_eval.sh 1000 \
#       --languages english russian simplified_chinese german japanese spanish french italian portuguese polish \
#       --exclude_samples results/statistics/cross_labeller/hplt-test-150/sample.parquet
#
# Output: results/statistics/cross_labeller/<SAMPLE_NAME>/ (sample.parquet, summary.json);
# SAMPLE_NAME defaults to hplt-test-<per_language>.
# CPU only: the python-data module (numpy, pyarrow) on the x86 CPU nodes.

set -euo pipefail

per_language="${1:-150}"
shift || true

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-data

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
output_dir="$base_dir/results/statistics/cross_labeller/${SAMPLE_NAME:-hplt-test-$per_language}"
echo "output_dir=$output_dir"

# Relative paths in the extra arguments resolve against the repository root.
cd "$base_dir"
python "$base_dir/scripts/data_processing/sample_cross_labeller_eval.py" \
    --topic_dataset "$base_dir/results/datasets/topic/hplt-finetune" \
    --format_dataset "$base_dir/results/datasets/format/hplt-finetune" \
    --split test \
    --per_language "$per_language" \
    --output_dir "$output_dir" \
    "$@"
echo "Done at $(date)"
