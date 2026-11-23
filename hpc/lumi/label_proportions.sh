#!/bin/bash
#SBATCH --job-name=label_proportions
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Count the topic and format proportions (top label, by documents and by characters, and
# the probability mass) of one finished language of the classifier output, per label and per
# topic x format pair. See scripts/data_analysis/label_proportions.py. Also checks that every
# input file has its output with one line per input document (hplt-sampler's counts).
#
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch label_proportions.sh deu_Latn
#
# Output: results/statistics/classifier_label_proportions/hplt-4.0-clean/<language>.json; the
# proportions are also printed as tables in the job's .err log.
# Speed: one worker reads ~44,000 documents (175 MB of text) per second; German (~1.05B
# documents) takes ~15-30 minutes with 32 workers, depending on the file system.

set -euo pipefail

usage="usage: sbatch label_proportions.sh <language, e.g. deu_Latn>"
language="${1:?$usage}"
[[ "$language" =~ ^[a-z]{3}_[A-Z][a-z]{3}$ ]] || { echo "Not an HPLT language code: '$language'; $usage" >&2; exit 1; }

source /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi/classify_hplt_common.sh

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Language: $language; output root: $output_root"

load_container_modules

srun singularity run -B /scratch/project_462001516 "$SIF" python \
    "$base_dir/scripts/data_analysis/label_proportions.py" \
    --input_dir "$output_root/$language" \
    --output "$base_dir/results/statistics/classifier_label_proportions/hplt-4.0-$collection/$language.json" \
    --expected_counts_dir "/scratch/project_462001516/users/tarkkaot/hplt-sampler/counts/v4/$language" \
    --workers "$SLURM_CPUS_PER_TASK"
