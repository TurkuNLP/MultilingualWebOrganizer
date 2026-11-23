#!/bin/bash
#SBATCH --job-name=classify_hplt_node
#SBATCH --account=project_462001516
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --gpus-per-node=8
#SBATCH -o ../../logs/%x_%A_%a.out
#SBATCH -e ../../logs/%x_%A_%a.err

# Label the HPLT 4.0 files of the given languages (default: those in classify_hplt_common.sh,
# deu_Latn spa_Latn fin_Latn) with the fine-tuned topic and format classifiers that take the
# URL, on full LUMI-G nodes (standard-g): each array task is one node, which labels up to 8
# files at a time, one per GPU (MI250X GCD).
# The same work as classify_hplt.sh, which runs one file per small-g task; both write the
# same output, so they can continue each other's work, but never run both on the same
# languages at once. See scripts/data_processing/classify_hplt.py.
#
#   sbatch --array=0-<nodes - 1> classify_hplt_node.sh <nodes> [language ...]
#   bash classify_hplt_node.sh list <nodes> [language ...]
#
# Files per node: the files are split into <nodes> fixed sets of about equal size in bytes
# (largest file first, each to the set with the fewest bytes so far), and array task K labels
# set K. On the node, the files run largest first, and a GPU starts the next file as soon as
# it finishes one, so GPUs are not left idle by uneven file sizes until the last 8 files.
# The sets depend only on <nodes>, the languages and their files, so a resubmission with the
# same <nodes> and languages gives every task the same files again. For a language with few
# files, or one much larger than the rest, classify_hplt.sh (one GPU per file) wastes less:
# here a node stays billed for 8 GPUs until its last file ends.
#
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   bash classify_hplt_node.sh list 10              # the files and gigabytes of each of 10 nodes
#   # Test first: 20,000 documents of each of the first 8 files of node 0 of 10, on dev-g;
#   # each file's log reports documents per second with 8 runs sharing the node.
#   TEST_DOCUMENTS=20000 sbatch --partition=dev-g --time=00:45:00 --array=0 classify_hplt_node.sh 10
#   # All files on 10 nodes; the array range must be 0-<nodes - 1>
#   sbatch --array=0-9 classify_hplt_node.sh 10
#
# Choosing <nodes>: a node labels about 8 x (documents per second of one run) documents per
# second; the three languages have about 2.2 billion documents, so all files take about
# 2.2e9 / (8 x rate x 3600) node hours, spread over the nodes. A task that reaches the 48-hour
# limit keeps its partial files: resubmit the same tasks with the same <nodes>
# (sbatch --array=<K,...> classify_hplt_node.sh <nodes>) and they continue; finished files
# are skipped at once. Do not change <nodes> while tasks of the run are still pending or
# running: the new sets would overlap with theirs. (A lock file per input still keeps two
# jobs from labelling the same file at once; the second stops with an error for that file.)
#
# Logs: the job's .out/.err (in ../../logs) list the files started and finished; each file's
# own log is in ../../logs/classify_hplt_node_<array job>_<task>/<language>_<shard>_<part>.log.
# A file that fails does not stop the others; the task exits with an error at the end and
# lists the failed files.

set -euo pipefail

source /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi/classify_hplt_common.sh

usage="usage: sbatch --array=0-<nodes - 1> classify_hplt_node.sh <nodes> [language ...], or bash classify_hplt_node.sh list <nodes> [language ...]"
mode="run"
if [[ "${1:-}" == "list" ]]; then
    mode="list"
    shift
fi
nodes="${1:?$usage}"
[[ "$nodes" =~ ^[1-9][0-9]*$ ]] || { echo "<nodes> must be a positive integer; $usage" >&2; exit 1; }
shift
set_languages "$@"
gpus_per_node=8

list_hplt_files
(( nodes <= ${#files[@]} )) || { echo "$nodes nodes for ${#files[@]} files; use at most ${#files[@]}" >&2; exit 1; }

# Split the files into the node sets: largest first, each to the set with the fewest bytes
# (ties: the lower set number; equal sizes: by path), so the split is deterministic.
mapfile -t by_size < <(for file in "${files[@]}"; do printf '%s %s\n' "$(stat -c %s "$file")" "$file"; done | LC_ALL=C sort -k1,1nr -k2,2)
set_bytes=()
set_files=()
for (( k = 0; k < nodes; k++ )); do set_bytes[k]=0; set_files[k]=""; done
for entry in "${by_size[@]}"; do
    size="${entry%% *}"
    file="${entry#* }"
    smallest=0
    for (( k = 1; k < nodes; k++ )); do
        if (( set_bytes[k] < set_bytes[smallest] )); then smallest=$k; fi
    done
    set_bytes[smallest]=$(( set_bytes[smallest] + size ))
    set_files[smallest]+="$file"$'\n'
done

if [[ "$mode" == "list" ]]; then
    for (( k = 0; k < nodes; k++ )); do
        mapfile -t node_files < <(printf '%s' "${set_files[k]}")
        echo "Node $k: ${#node_files[@]} files, $(( set_bytes[k] / 1000000000 )) GB"
        printf '    %s\n' "${node_files[@]}"
    done
    echo "${#files[@]} files of ${languages[*]} on $nodes nodes: submit with" \
        "sbatch --array=0-$(( nodes - 1 )) classify_hplt_node.sh $nodes${*:+ $*}"
    exit 0
fi

task="${SLURM_ARRAY_TASK_ID:?Submit as an array job; $usage}"
(( task < nodes )) || { echo "Task $task, but <nodes> is $nodes (tasks 0-$(( nodes - 1 )))" >&2; exit 1; }
mapfile -t node_files < <(printf '%s' "${set_files[task]}")
if [[ -n "${TEST_DOCUMENTS:-}" ]]; then
    # One file per GPU is enough to measure the speed; without this, every file of the set
    # would get its first TEST_DOCUMENTS documents labelled.
    node_files=("${node_files[@]:0:gpus_per_node}")
fi

echo "Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}_$task, node set $task of $nodes) on $(hostname) at $(date)"
echo "Languages: ${languages[*]}"
echo "${#node_files[@]} files, $(( set_bytes[task] / 1000000000 )) GB in all; $gpus_per_node at a time"
echo "Output root: $output_root"
command -v squeue > /dev/null || { echo "squeue not found; it is needed to check for stale locks" >&2; exit 1; }

load_container_modules

export PYTHONWARNINGS="ignore::FutureWarning"
export TRANSFORMERS_VERBOSITY="error"

extra_args=()
if [[ -n "${TEST_DOCUMENTS:-}" ]]; then
    echo "Test run: the first $gpus_per_node files, stopping after $TEST_DOCUMENTS documents each"
    extra_args+=(--max_documents "$TEST_DOCUMENTS")
fi

run_dir="$base_dir/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${task}"
mkdir -p "$run_dir"
echo "File logs: $run_dir"

# Name of an input's log and status files: <language>_<shard>_<part>.
file_key() {
    echo "$(basename "$(dirname "$1")")_$(basename "$1" .jsonl.zst)"
}

# Label one file in the background, as a job step on one GPU and 7 cores. The step's exit
# status is written to <key>.status when it ends. srun reads the job's SLURM_GPUS_PER_NODE=8
# as its own --gpus-per-node, so the step overrides it with 1; otherwise one step could take
# all 8 GPUs and the others would wait. --exact keeps each step to its own cores and memory.
start_file() {
    local input="$1"
    local key
    key=$(file_key "$input")
    remove_stale_lock "$input"
    rm -f -- "$run_dir/$key.status"
    (
        status=0
        srun --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=60G --exact \
            singularity run -B /scratch/project_462001516 -B /appl/local/openeurollm "$SIF" python "$python_script" \
            --input "$input" \
            --output_root "$output_root" \
            --topic_model "$topic_model" \
            --format_model "$format_model" \
            --max_length "$max_length" \
            "${extra_args[@]}" \
            >> "$run_dir/$key.log" 2>&1 || status=$?
        echo "$status" > "$run_dir/$key.status"
    ) &
    running[$!]="$input"
    echo "$(date '+%F %T') started $input"
}

declare -A running=()
failed=()
next=0
while (( next < ${#node_files[@]} || ${#running[@]} > 0 )); do
    while (( ${#running[@]} < gpus_per_node && next < ${#node_files[@]} )); do
        start_file "${node_files[next]}"
        next=$(( next + 1 ))
    done
    sleep 30
    for pid in "${!running[@]}"; do
        kill -0 "$pid" 2> /dev/null && continue
        input="${running[$pid]}"
        unset "running[$pid]"
        status_file="$run_dir/$(file_key "$input").status"
        status=$(cat "$status_file" 2> /dev/null || echo "none (the step was killed)")
        if [[ "$status" == "0" ]]; then
            echo "$(date '+%F %T') finished $input"
        else
            echo "$(date '+%F %T') FAILED $input (exit status $status; log $run_dir/$(file_key "$input").log)" >&2
            failed+=("$input")
        fi
    done
done

echo "Done at $(date): $(( ${#node_files[@]} - ${#failed[@]} )) of ${#node_files[@]} files ended without error"
if (( ${#failed[@]} > 0 )); then
    echo "${#failed[@]} files failed:" >&2
    printf '    %s\n' "${failed[@]}" >&2
    exit 1
fi
