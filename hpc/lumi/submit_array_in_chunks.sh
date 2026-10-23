#!/bin/bash
# Submit a large Slurm array in smaller pieces, waiting whenever the submit limit is reached.
# Slurm counts every array task against your submit limit (210 on LUMI small-g), and pending
# tasks count too, so an array bigger than the free room is rejected outright. This script
# submits chunk_size tasks at a time and, when sbatch is rejected for the submit limit
# (AssocMaxSubmitJobLimit), waits and tries the same chunk again. Any other error stops it.
#
# Run it on a login node in tmux, not with sbatch, from the directory you would submit from:
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   tmux new -s labelling        # note the login node (hostname) to reattach later
#   bash submit_array_in_chunks.sh <task_ids> <chunk_size> <running_cap> <sbatch args...> <script> [script args]
# e.g.
#   bash submit_array_in_chunks.sh 0-53,55-95,97-102,104-127,128-181,183-223,225-230,232-255 40 36 \
#       --time=12:00:00 array_generate_distribution_hplt.sh
#
# task_ids: the tasks to submit, as for --array (ranges and commas, no %).
# running_cap: how many tasks of one chunk may run at once (the %N of --array).
# The submitted job IDs and their tasks are printed as they go.

set -euo pipefail

usage="usage: bash submit_array_in_chunks.sh <task_ids> <chunk_size> <running_cap> <sbatch args...> <script> [script args]"
task_ids="${1:?$usage}"
chunk_size="${2:?$usage}"
running_cap="${3:?$usage}"
shift 3
(( $# >= 1 )) || { echo "$usage" >&2; exit 1; }
for value in "$chunk_size" "$running_cap"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "chunk_size and running_cap must be positive integers; $usage" >&2; exit 1; }
done
wait_seconds=600

# Expand "0-3,7,9-10" into 0 1 2 3 7 9 10.
tasks=()
IFS=, read -r -a parts <<< "$task_ids"
for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        for (( i = BASH_REMATCH[1]; i <= BASH_REMATCH[2]; i++ )); do tasks+=("$i"); done
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
        tasks+=("$part")
    else
        echo "Cannot read task IDs '$part' in '$task_ids'; $usage" >&2
        exit 1
    fi
done

# Write a list of task IDs back as ranges: 0 1 2 3 7 -> 0-3,7.
compress() {
    local out="" start="" previous=""
    for i in "$@"; do
        if [[ -z "$start" ]]; then
            start=$i
        elif (( i != previous + 1 )); then
            out+="${out:+,}$start$([[ $previous != "$start" ]] && echo "-$previous")"
            start=$i
        fi
        previous=$i
    done
    [[ -n "$start" ]] && out+="${out:+,}$start$([[ $previous != "$start" ]] && echo "-$previous")"
    echo "$out"
}

echo "Submitting ${#tasks[@]} tasks in chunks of $chunk_size on $(hostname) at $(date)"
for (( first = 0; first < ${#tasks[@]}; first += chunk_size )); do
    spec=$(compress "${tasks[@]:first:chunk_size}")
    while true; do
        if output=$(sbatch --parsable --array="$spec%$running_cap" "$@" 2>&1); then
            echo "$(date '+%F %T') submitted job ${output%%;*}: tasks $spec"
            break
        fi
        if [[ "$output" == *AssocMaxSubmitJobLimit* || "$output" == *QOSMaxSubmitJob* ]]; then
            echo "$(date '+%F %T') submit limit reached; trying tasks $spec again in $((wait_seconds / 60)) minutes"
            sleep "$wait_seconds"
        else
            echo "sbatch failed for tasks $spec:" >&2
            echo "$output" >&2
            exit 1
        fi
    done
done
echo "All ${#tasks[@]} tasks submitted at $(date)"
