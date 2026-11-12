#!/bin/bash
#SBATCH --job-name=find_duplicates
#SBATCH --account=project_2020507
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Find exact and near-duplicate documents in a dataset made by create_datasets.sh, within
# the training split and between the training split and the validation and test splits,
# with scripts/data_analysis/find_duplicates.py (character n-gram MinHash + LSH).
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch find_duplicates.sh <label type>/<dataset name> [find_duplicates.py args]
# e.g.
#   sbatch find_duplicates.sh topic/hplt-finetune
# The topic, format and -nourl datasets hold the same documents and texts, so one of them
# is enough. A quick test on the first 20,000 documents of each split:
#   sbatch --partition=test --time=00:15:00 find_duplicates.sh topic/hplt-finetune \
#       --max_documents 20000 --output_dir ../../results/statistics/duplicates/test
#
# Output: results/statistics/duplicates/<label type>/<dataset name>/ (summary.json,
# languages.csv, duplicates.tsv, examples.jsonl), unless --output_dir is given.
# CPU only: runs on the (x86) CPU nodes with the python-data module, which has numpy and
# pyarrow (all the script needs besides the standard library). Submit from a Roihu-CPU
# login node. Memory: about 4 GB of MinHash signatures for 7.3M documents, plus the
# workers' batches of text.
# Expected time for 7.3M documents with 63 workers: about 10-20 minutes, mostly reading
# the texts.

set -euo pipefail

usage="usage: sbatch find_duplicates.sh <label type>/<dataset name> [find_duplicates.py args]"
dataset_name="${1:?$usage}"
shift

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-data

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_analysis/find_duplicates.py"
dataset_dir="$base_dir/results/datasets/$dataset_name"
output_dir="$base_dir/results/statistics/duplicates/$dataset_name"

if [[ ! -f "$dataset_dir/dataset_dict.json" ]]; then
    echo "No dataset at $dataset_dir (expected <label type>/<dataset name>, e.g. topic/hplt-finetune)" >&2
    exit 1
fi
echo "Dataset: $dataset_dir"
echo "Arguments: $*"

# Relative paths in the arguments are relative to this directory (hpc/roihu). A later
# --output_dir in the arguments replaces the default.
python "$python_script" \
    --dataset "$dataset_dir" \
    --output_dir "$output_dir" \
    --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 1 )) \
    "$@"
