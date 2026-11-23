#!/bin/bash
# Wait until a classify_hplt_node.sh (or classify_hplt.sh) array has left the queue, then
# submit every file it left unfinished to small-g with classify_hplt.sh, one GPU per file,
# each with a time limit fitted to its remaining work. Unfinished files continue from their
# partial output.
#
# Run it on a login node in tmux, not with sbatch (note the login node to reattach later):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   tmux new -s leftovers
#   bash submit_classify_leftovers.sh <array job ID> [language ...]
# e.g. bash submit_classify_leftovers.sh 23784568
#
# The languages (default: those in classify_hplt_common.sh, deu_Latn spa_Latn fin_Latn) must be
# those of the array that is waited for, as they decide which files are checked.
#
# Waiting: squeue is asked every 10 minutes whether any task of the array is still in the
# queue (in any state but an end state such as COMPLETED or TIMEOUT). An squeue failure is
# never taken to mean that the array has ended; it is retried.
#
# Time limits: the remaining documents of each file (its document count, from hplt-sampler's
# per-file counts, minus the lines of its partial output) at a cautious rate per language
# (below), plus 10% and an hour for loading the models, resuming, compressing and checking,
# rounded up to 15 minutes and at most small-g's 72 hours. The tasks are submitted as one
# array with the longest limit, and each task's limit is then lowered to its own, which
# a user may do for pending tasks. If a file's remaining work cannot be estimated, it gets
# the fallback limit and the log says so.
#
# Everything is also written to ../../logs/submit_classify_leftovers_<array job ID>_<time>.log.

set -euo pipefail

source /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi/classify_hplt_common.sh

usage="usage: bash submit_classify_leftovers.sh <array job ID> [language ...]"
job="${1:?$usage}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo "Not a job ID: '$job'; $usage" >&2; exit 1; }
shift
set_languages "$@"

poll_seconds=600
# Documents per second per GPU, for the time limits. Cautious: the tokenizer-based estimates
# from the random clean samples (1,180 tokens per document for Hungarian at the ~74,500 tokens
# per second measured on LUMI), which were 5-10% below the rates measured for deu, spa and fin.
declare -A rates=([deu_Latn]=80 [spa_Latn]=87 [fin_Latn]=66 [hun_Latn]=63)
default_rate=60
fallback_minutes=$(( 48 * 60 ))
max_minutes=$(( 72 * 60 ))  # small-g's longest time limit
counts_dir="/scratch/project_462001516/users/tarkkaot/hplt-sampler/counts/v4"  # clean collection

# sbatch resolves classify_hplt.sh's relative log paths from the directory it runs in.
cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
log_file="$base_dir/logs/submit_classify_leftovers_${job}_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1

log() {
    echo "$(date '+%F %T') $*"
}

# Format minutes as a SLURM time limit, H:MM:00.
time_limit() {
    printf '%d:%02d:00' $(( $1 / 60 )) $(( $1 % 60 ))
}

# Return 0 if a task of the array is still in the queue (any state but an end state), 1 if
# none is, 2 if squeue failed.
array_in_queue() {
    local out state
    if ! out=$(squeue -h -j "$job" -o %T 2>&1); then
        # A job that has left the controller's memory is unknown to squeue.
        [[ "$out" == *"Invalid job id"* ]] && return 1
        log "squeue failed ($out); trying again in $(( poll_seconds / 60 )) minutes" >&2
        return 2
    fi
    while read -r state; do
        case "$state" in
            ""|COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) ;;
            *) return 0 ;;
        esac
    done <<< "$out"
    return 1
}

# Print the time limit in minutes for an input file's remaining work, or return 1 if it
# cannot be estimated (no count for the file, or a partial output longer than the count).
estimate_minutes() {
    local input="$1"
    local language name count_file count done_lines remaining rate
    [[ "$collection" == "clean" ]] || return 1
    language=$(basename "$(dirname "$input")")
    name=$(basename "$input" .jsonl.zst)
    count_file="$counts_dir/$language/$name.jsonl.zst.json"
    [[ -f "$count_file" ]] || return 1
    count=$(sed -n 's/.*"count": *\([0-9][0-9]*\).*/\1/p' "$count_file")
    [[ -n "$count" ]] || return 1
    done_lines=0
    if [[ -f "$output_root/$language/$name.jsonl.partial" ]]; then
        done_lines=$(wc -l < "$output_root/$language/$name.jsonl.partial")
    fi
    (( done_lines <= count )) || return 1
    remaining=$(( count - done_lines ))
    rate="${rates[$language]:-$default_rate}"
    awk -v remaining="$remaining" -v rate="$rate" -v cap="$max_minutes" -v count="$count" -v done="$done_lines" 'BEGIN {
        minutes = remaining / rate / 60 * 1.1 + 60
        quarters = int(minutes / 15); if (quarters * 15 < minutes) quarters++
        minutes = quarters * 15; if (minutes > cap) minutes = cap
        printf "%d %d %d\n", minutes, done, count
    }'
}

# Submit the array, waiting and retrying while the submit limit is reached. Prints the job ID.
submit_array() {
    local spec="$1" limit="$2" out
    while true; do
        if out=$(sbatch --parsable --array="$spec" --time="$limit" classify_hplt.sh "${languages[@]}" 2>&1); then
            echo "${out%%;*}"
            return 0
        fi
        if [[ "$out" == *AssocMaxSubmitJobLimit* || "$out" == *QOSMaxSubmitJob* ]]; then
            log "Submit limit reached; trying again in 10 minutes" >&2
            sleep 600
        else
            log "sbatch failed: $out" >&2
            return 1
        fi
    done
}

log "Watching array $job on $(hostname) for ${languages[*]}; log $log_file"
command -v squeue > /dev/null || { log "squeue not found" >&2; exit 1; }
while true; do
    status=0
    array_in_queue || status=$?
    if (( status == 1 )); then
        break
    fi
    if (( status == 0 )); then
        log "Array $job still has tasks in the queue; checking again in $(( poll_seconds / 60 )) minutes"
    fi
    sleep "$poll_seconds"
done
log "Array $job has left the queue"
# Let the last tasks' file operations settle on the file system.
sleep 60

list_hplt_files
list_pending_tasks
if (( ${#pending[@]} == 0 )); then
    log "All ${#files[@]} files of ${languages[*]} are finished; nothing to submit"
    exit 0
fi
log "${#pending[@]} of ${#files[@]} files are not finished; estimating their remaining work"

declare -A task_minutes=()
longest=0
for i in "${pending[@]}"; do
    input="${files[$i]}"
    if estimate=$(estimate_minutes "$input"); then
        read -r minutes done_lines count <<< "$estimate"
        log "Task $i: $input: $done_lines of $count documents done; limit $(time_limit "$minutes")"
    else
        minutes=$fallback_minutes
        log "Task $i: $input: cannot estimate the remaining work; fallback limit $(time_limit "$minutes")" >&2
    fi
    lock=$(lock_path "$input")
    if [[ -f "$lock" ]]; then
        log "Task $i: $input still has a lock ($(tr '\n' ' ' < "$lock")); the task removes it if its job has ended" >&2
    fi
    task_minutes[$i]=$minutes
    if (( minutes > longest )); then
        longest=$minutes
    fi
done

spec=$(IFS=,; echo "${pending[*]}")
log "Submitting: sbatch --array=$spec --time=$(time_limit "$longest") classify_hplt.sh ${languages[*]}"
new_job=$(submit_array "$spec" "$(time_limit "$longest")")
log "Submitted array $new_job"

# Lower each task's limit to its own estimate while the tasks are pending.
for i in "${pending[@]}"; do
    minutes="${task_minutes[$i]}"
    (( minutes < longest )) || continue
    if scontrol update JobId="${new_job}_$i" TimeLimit="$(time_limit "$minutes")"; then
        log "Task ${new_job}_$i: time limit $(time_limit "$minutes")"
    else
        log "Could not lower the time limit of ${new_job}_$i; it keeps $(time_limit "$longest")" >&2
    fi
done
log "Done. Check with: squeue --me -r -j $new_job -o \"%i %l %T\""
