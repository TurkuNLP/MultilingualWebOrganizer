#!/bin/bash
#SBATCH --job-name=check_mixture_data
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Check the tokenized training data of a mixture (made by build_mixture_data.sh) against
# the planned mixture: counts the tokens of every topic x format pair from the Megatron .idx
# files and the selection's .labels.tsv files, and fails if the training split (99% of the
# documents) holds fewer than 100B tokens. See scripts/data_analysis/check_mixture_data.py.
#
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch check_mixture_data.sh deu_Latn [mixture]
# mixture defaults to "both".
#
# Output: results/mixtures/hplt-4.0-clean/<language>/<mixture>.tokens.json; the planned and
# realized topic and format token shares are printed as tables in the job's .err log.
# Single-threaded; it reads the labels files (~7 GB for German) once.

set -euo pipefail

usage="usage: sbatch check_mixture_data.sh <language, e.g. deu_Latn> [mixture]"
language="${1:?$usage}"
[[ "$language" =~ ^[a-z]{3}_[A-Z][a-z]{3}$ ]] || { echo "Not an HPLT language code: '$language'; $usage" >&2; exit 1; }
mixture_name="${2:-both}"

source /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi/classify_hplt_common.sh

mixture="$base_dir/results/mixtures/hplt-4.0-$collection/$language/$mixture_name.json"
data_root="$base_dir/data/megatron/oellm-v1-256k/hplt-4.0-$collection-$mixture_name"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Language: $language; mixture: $mixture; data: $data_root/$language"

load_container_modules

srun singularity run -B /scratch/project_462001516 "$SIF" python \
    "$base_dir/scripts/data_analysis/check_mixture_data.py" \
    --mixture "$mixture" \
    --megatron_dir "$data_root/$language" \
    --selection_dir "$data_root/selection/$language" \
    --output "$base_dir/results/mixtures/hplt-4.0-$collection/$language/$mixture_name.tokens.json"

echo "Done at $(date)"
