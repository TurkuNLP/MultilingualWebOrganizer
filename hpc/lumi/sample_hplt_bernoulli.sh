#!/bin/bash
#SBATCH --job-name=sample_hplt_bernoulli
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -o ../../logs/%x_%A_%a.out
#SBATCH -e ../../logs/%x_%A_%a.err
#SBATCH --array=0-255%16

# Sample exactly 30,000 HPLT 4.0 documents per language at random: every document of every
# file is kept with the same probability, set to draw slightly more than 30,000; the drawn
# documents are shuffled and the first 30,000 kept.
# Unlike sample_hplt_by_quality.sh, there are no quotas per quality bucket, and the whole
# of every file is read (twice: once to count, once to sample).
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   sbatch sample_hplt_bernoulli.sh [extra sample_hplt_by_quality.py args]
# e.g. sbatch --array=0,128 sample_hplt_bernoulli.sh          (Afrikaans, clean and noisy)
#      sbatch sample_hplt_bernoulli.sh --ratio 0.001           (a fixed ratio, one pass)
#
# One task per language of data/qwen_label_languages.txt and collection:
#   tasks 0-127:   clean (language N of the file is task N)
#   tasks 128-255: noisy (language N of the file is task 128 + N)
# If the languages file changes, also change --array (2 * the number of languages).
# The "%16" limit runs at most 16 tasks at once, to spare the file system.
#
# Each task reads its files with one worker per CPU. Large languages read the most data;
# the time limit is a guess, so check the first tasks before submitting the rest.
#
# Output: data/hplt_<collection>_random_sample/<language>.jsonl and <language>.manifest.json.
# A language whose output exists is skipped unless --overwrite is given.

set -euo pipefail

echo "Job ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-} on $(hostname) at $(date)"

module purge
module use /appl/local/laifs/modules
module load Local-LAIF lumi-aif-singularity-bindings

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/sample_hplt_by_quality.py"
SIF="$base_dir/environments/lumi/vllm.sif"
languages_file="$base_dir/data/qwen_label_languages.txt"
collections=(clean noisy)

# Language names: the part of each non-empty line before the colon.
mapfile -t languages < <(sed -n 's/^[[:space:]]*\([^:]*[^:[:space:]]\)[[:space:]]*:.*/\1/p' "$languages_file")
num_languages=${#languages[@]}
if (( SLURM_ARRAY_TASK_ID >= num_languages * ${#collections[@]} )); then
    echo "Array task $SLURM_ARRAY_TASK_ID is out of range for $num_languages languages and ${#collections[@]} collections" >&2
    exit 1
fi
language=${languages[$((SLURM_ARRAY_TASK_ID % num_languages))]}
collection=${collections[$((SLURM_ARRAY_TASK_ID / num_languages))]}

input_dir="/appl/local/openeurollm/training/catalogue/hplt/4.0/${collection}"
output_dir="$base_dir/data/hplt_${collection}_random_sample"
echo "Language:   $language"
echo "Collection: $collection"
echo "Input:      $input_dir"
echo "Output:     $output_dir"

srun singularity run -B /scratch/project_462001516 -B /appl/local/openeurollm "$SIF" python "$python_script" \
    --method bernoulli \
    --languages-file "$languages_file" \
    --languages "$language" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --sample-size 50000 \
    --workers "$SLURM_CPUS_PER_TASK" \
    "$@"
