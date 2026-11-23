#!/bin/bash
#SBATCH --job-name=create_weborganizer_eval_dataset
#SBATCH --account=project_2020507
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Turn WebOrganizer's Llama-3.1-405B annotations (validation and test splits, 10K English
# pages each) into evaluation datasets in our format, with
# scripts/data_processing/create_weborganizer_eval_dataset.py. Writes four datasets:
#   results/datasets/{topic,format}/weborganizer-405B        URL and text
#   results/datasets/{topic,format}/weborganizer-405B-nourl  text only
# Targets: Llama's 24-label distribution, then 0 for Other and Bad example (our label order).
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch create_weborganizer_eval_dataset.sh [--overwrite]
#
# With TARGETS=qwen, the targets are instead our labeller's 26-label distributions from
# results/distribution/Qwen3.8-27B-weborganizer-annotations/<label type>/<split>.jsonl (made by
# array_label_weborganizer_eval.sh), and the datasets are weborganizer-qwen and
# weborganizer-qwen-nourl. Only the validation split is labelled by default, so:
#   TARGETS=qwen sbatch create_weborganizer_eval_dataset.sh
# SPLITS sets the splits (default "validation test" for Llama targets, "validation" for Qwen).
# Downloads the two Hugging Face datasets (about 300 MB each) into the cache set in .bashrc.
# The job needs no GPU, but the python-vllm module (which has `datasets`, the same version
# finetune.py reads the datasets with) only runs on the (ARM) GPU nodes. The log lists which
# name in Llama's prompt each label index had; check that they match the label names.

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/create_weborganizer_eval_dataset.py"

targets="${TARGETS:-llama}"
case "$targets" in
    llama) name=weborganizer-405B; default_splits="validation test" ;;
    qwen) name=weborganizer-qwen; default_splits="validation" ;;
    *) echo "TARGETS must be llama or qwen, not $targets" >&2; exit 1 ;;
esac
read -r -a splits <<< "${SPLITS:-$default_splits}"
echo "targets=$targets splits=${splits[*]}"

for label_type in topic format; do
    target_args=()
    if [[ "$targets" == qwen ]]; then
        target_args=(--qwen_labels_dir "$base_dir/results/distribution/Qwen3.8-27B-weborganizer-annotations/$label_type")
    fi
    for variant in "" "-nourl"; do
        output_dir="$base_dir/results/datasets/$label_type/$name$variant"
        echo "== $label_type -> $output_dir"
        no_url_args=()
        [[ -n "$variant" ]] && no_url_args=(--no_url)
        python "$python_script" \
            --label_type "$label_type" \
            --output_dir "$output_dir" \
            --reference_dataset "$base_dir/results/datasets/$label_type/hplt-finetune" \
            --splits "${splits[@]}" \
            "${target_args[@]}" \
            "${no_url_args[@]}" \
            "$@"
    done
done
echo "Done at $(date)"
