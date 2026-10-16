#!/bin/bash
#SBATCH --job-name=sample_hplt
#SBATCH --account=project_462001516
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Sample 100,000 HPLT 4.0 documents per language, stratified by quality bucket, from the
# collection set in data_quality below (clean, noisy or poor).
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch sample_hplt_by_quality.sh [extra sample_hplt_by_quality.py args]
# e.g. sbatch sample_hplt_by_quality.sh --languages Maltese Norwegian
#      sbatch sample_hplt_by_quality.sh --overwrite
#
# Reads the languages and their language codes from data/oellm_languages.txt and
# writes data/hplt_<data_quality>_sample/<language>.jsonl and <language>.manifest.json.
# Languages whose output file exists are skipped with a warning unless --overwrite is given.
# Languages and language codes without a directory in the HPLT catalogue are
# reported as warnings in the error log; the job still succeeds.
# Only the head of each file is read, so the job needs little time and memory.

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/sample_hplt_by_quality.py"
SIF="$base_dir/environments/lumi/vllm.sif"

data_quality=poor # clean, noisy or poor
input_dir="/appl/local/openeurollm/training/catalogue/hplt/4.0/${data_quality}"
languages_file="$base_dir/data/qwen_label_languages.txt"
output_dir="$base_dir/data/hplt_${data_quality}_sample"
echo "Input:     $input_dir"
echo "Languages: $languages_file"
echo "Output:    $output_dir"

srun singularity run -B /scratch/project_462001516 -B /appl/local/openeurollm "$SIF" python "$python_script" \
    --languages-file "$languages_file" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --sample-size 10000 \
    "$@"
