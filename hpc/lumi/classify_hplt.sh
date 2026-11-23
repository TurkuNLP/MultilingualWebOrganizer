#!/bin/bash
#SBATCH --job-name=classify_hplt
#SBATCH --account=project_462001516
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH -o ../../logs/%x_%A_%a.out
#SBATCH -e ../../logs/%x_%A_%a.err

# Label every HPLT 4.0 file of the given languages (default: those in classify_hplt_common.sh,
# deu_Latn spa_Latn fin_Latn) with the fine-tuned topic and format classifiers that take the
# URL (mmbert-{topic,format}-1, not the -nourl ones), one array task per file, on one GPU (one
# MI250X GCD) each.
# See scripts/data_processing/classify_hplt.py. For full nodes of 8 GPUs on standard-g, use
# classify_hplt_node.sh; both write the same output, so they can continue each other's work,
# but never run both on the same languages at once.
#
#   sbatch --array=<tasks> classify_hplt.sh [language ...]
#   bash classify_hplt.sh list [language ...]
#   bash classify_hplt.sh pending [language ...]   # tasks whose file is not finished yet
#
# Task N labels file N of the list of all <shard>_<part>.jsonl.zst files of the languages, in
# language order, then by shard and part, so a run's task numbers depend on its languages:
# resubmit tasks with the same languages. Print the list and the number of tasks with
#   bash classify_hplt.sh list [language ...]
#
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   bash classify_hplt.sh list                     # file list; the last line gives the array range
#   # Another language, e.g. Hungarian (15 files): one task per file, 48 hours for the largest
#   bash classify_hplt.sh list hun_Latn
#   sbatch --array=0-14 --time=48:00:00 classify_hplt.sh hun_Latn
#   # Test first: 20,000 documents of task 0 on dev-g; the log reports documents per second.
#   TEST_DOCUMENTS=20000 sbatch --partition=dev-g --time=00:30:00 --array=0 classify_hplt.sh
#   # All files: 326 tasks, more than small-g's submit limit of 210 tasks (pending ones count),
#   # so submit them in chunks of 100 on a login node in tmux; a chunk that hits the limit is
#   # retried every 10 minutes. The running cap (20) is per chunk, and two chunks fit in the
#   # limit at once, so up to 40 tasks run at a time.
#   bash submit_array_in_chunks.sh 0-325 100 20 classify_hplt.sh
#
# Output: results/classifier_predictions/mmbert-1/hplt-4.0-<collection>/<language>/
# <shard>_<part>.jsonl.zst, with metadata.json (models, label order, settings) next to the
# language directories. A task whose file is finished exits at once, and a task that was cut
# off (time limit, node failure) continues its file when resubmitted, so resubmitting the same
# array is safe. The test run's documents are kept and continued by the full run.
# A lock file per input keeps two jobs from labelling the same file at once.

set -euo pipefail

source /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi/classify_hplt_common.sh

mode="run"
if [[ "${1:-}" == "list" || "${1:-}" == "pending" ]]; then
    mode="$1"
    shift
fi
set_languages "$@"
list_hplt_files

if [[ "$mode" == "list" ]]; then
    for i in "${!files[@]}"; do echo "$i ${files[$i]}"; done
    echo "${#files[@]} files of ${languages[*]}: submit with --array=0-$(( ${#files[@]} - 1 ))" \
        "classify_hplt.sh${*:+ $*}"
    exit 0
fi

if [[ "$mode" == "pending" ]]; then
    # The tasks whose file has no finished output yet, e.g. to finish on small-g the files a
    # classify_hplt_node.sh run left at its time limit. A file whose lock is held is being
    # labelled by a job; check that no job still runs before submitting it.
    list_pending_tasks
    for i in "${pending[@]}"; do
        lock=$(lock_path "${files[$i]}")
        if [[ -f "$lock" ]]; then
            echo "$i ${files[$i]} (locked: $(tr '\n' ' ' < "$lock"))"
        else
            echo "$i ${files[$i]}"
        fi
    done
    if (( ${#pending[@]} == 0 )); then
        echo "All ${#files[@]} files of ${languages[*]} are finished"
    else
        echo "${#pending[@]} of ${#files[@]} files of ${languages[*]} are not finished: submit with" \
            "--array=$(IFS=,; echo "${pending[*]}") classify_hplt.sh${*:+ $*}"
    fi
    exit 0
fi

task="${SLURM_ARRAY_TASK_ID:?Submit as an array job (sbatch --array=...), or run 'bash classify_hplt.sh list'}"
(( task < ${#files[@]} )) || { echo "Task $task, but there are only ${#files[@]} files (0-$(( ${#files[@]} - 1 )))" >&2; exit 1; }
input="${files[$task]}"

echo "Job ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-}_$task) on $(hostname) at $(date)"
echo "Languages: ${languages[*]}"
echo "Input: $input"
echo "Output root: $output_root"

load_container_modules
remove_stale_lock "$input"

export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

extra_args=()
if [[ -n "${TEST_DOCUMENTS:-}" ]]; then
    echo "Test run: stopping after $TEST_DOCUMENTS documents"
    extra_args+=(--max_documents "$TEST_DOCUMENTS")
fi

srun singularity run -B /scratch/project_462001516 -B /appl/local/openeurollm "$SIF" python "$python_script" \
    --input "$input" \
    --output_root "$output_root" \
    --topic_model "$topic_model" \
    --format_model "$format_model" \
    --max_length "$max_length" \
    "${extra_args[@]}"
