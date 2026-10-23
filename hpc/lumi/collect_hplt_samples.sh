#!/bin/bash
#SBATCH --job-name=collect_hplt_samples
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Collect the hplt-sampler runs submitted by submit_hplt_sampler.sh into one sample file
# per language of data/qwen_label_languages.txt: data/hplt_<collection>_random_sample/
# <language>.jsonl, 30,000 documents each (all of them for a language with fewer), where
# reuse_hplt_labels.sh and the labelling script read them.
# How to run on LUMI (submit from this directory so relative log paths resolve), once the
# runs of a collection have finished:
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch collect_hplt_samples.sh <clean|noisy|poor> [extra collect_hplt_samples.py args]
# e.g. sbatch collect_hplt_samples.sh clean --allow-incomplete   (collect the finished ones)
#      sbatch collect_hplt_samples.sh noisy --languages Maltese Serbian
#
# Stops on an unfinished run unless --allow-incomplete is given, and on a run made without
# shuffling or a repeated document ID. Existing sample files are kept unless --overwrite
# is given, so rerunning with --allow-incomplete adds the languages finished since.

set -euo pipefail

collection="${1:?usage: sbatch collect_hplt_samples.sh <clean|noisy|poor> [extra args]}"
shift

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/collect_hplt_samples.py"
SIF="$base_dir/environments/lumi/vllm.sif"
sampler_dir="/scratch/project_462001516/users/tarkkaot/hplt-sampler"
output_dir="$base_dir/data/hplt_${collection}_random_sample"
echo "Collection: $collection"
echo "Runs:       $sampler_dir"
echo "Output:     $output_dir"

srun singularity run -B /scratch/project_462001516 -B /appl/local/openeurollm "$SIF" python "$python_script" \
    --collection "$collection" \
    --sampler-dir "$sampler_dir" \
    --languages-file "$base_dir/data/qwen_label_languages.txt" \
    --output-dir "$output_dir" \
    --seed 42 \
    --n-samples 30000 \
    --allow-fewer \
    "$@"
