# Settings and helpers shared by classify_hplt.sh (one GPU per task, small-g) and
# classify_hplt_node.sh (full nodes of 8 GPUs, standard-g), so both label the same files into
# the same output with the same models. Sourced by them, not run on its own.

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"

collection="clean"
# The default languages, used when a launcher is given none. The file list, and so the task
# numbers and node sets, depend on the languages: resubmit a run with the same languages.
default_languages=(deu_Latn spa_Latn fin_Latn)
languages=("${default_languages[@]}")
hplt_dir="/appl/local/openeurollm/training/catalogue/hplt/4.0/$collection"

python_script="$base_dir/scripts/data_processing/classify_hplt.py"
SIF="$base_dir/environments/lumi/vllm.sif"
topic_model="$base_dir/results/models/topic/hplt-finetune/mmbert-topic-1"
format_model="$base_dir/results/models/format/hplt-finetune/mmbert-format-1"
output_root="$base_dir/results/classifier_predictions/mmbert-1/hplt-4.0-$collection"

# Tokens per document (URL and text); longer inputs are truncated. 8192, as the models were
# trained and evaluated. 4096 would change the input of about 1.5% of the documents (those
# longer than 4096 tokens) and cut only about 5% of the tokens to label, by an estimate from
# the first 20,000 documents of 8 German and Spanish files. It is recorded in the output
# root's metadata.json, and a run with another value there stops with an error.
max_length=8192

# Set the global array "languages" to the given HPLT language codes (e.g. hun_Latn), or to
# the defaults if none are given. Exits on a code that is not of the form xxx_Xxxx, or one
# given twice.
set_languages() {
    local language
    if (( $# == 0 )); then
        languages=("${default_languages[@]}")
        return 0
    fi
    for language in "$@"; do
        [[ "$language" =~ ^[a-z]{3}_[A-Z][a-z]{3}$ ]] || { echo "Not an HPLT language code: '$language' (e.g. hun_Latn)" >&2; exit 1; }
    done
    if (( $(printf '%s\n' "$@" | sort -u | wc -l) != $# )); then
        echo "A language is given twice: $*" >&2
        exit 1
    fi
    languages=("$@")
}

# Fill the global array "files" with every <shard>_<part>.jsonl.zst file of the languages, in
# language order, then by shard and part (5_1, 5_2, ..., 10_1). Exits on a missing language
# directory, an empty one, or an unexpected file name.
list_hplt_files() {
    files=()
    local language language_dir name names
    for language in "${languages[@]}"; do
        language_dir="$hplt_dir/$language"
        [[ -d "$language_dir" ]] || { echo "No directory $language_dir" >&2; exit 1; }
        mapfile -t names < <(find "$language_dir" -maxdepth 1 -type f -name '*.jsonl.zst' -printf '%f\n' | LC_ALL=C sort -t_ -k1,1n -k2,2n)
        (( ${#names[@]} > 0 )) || { echo "No .jsonl.zst files in $language_dir" >&2; exit 1; }
        for name in "${names[@]}"; do
            [[ "$name" =~ ^[0-9]+_[0-9]+\.jsonl\.zst$ ]] || { echo "Unexpected file name $language_dir/$name" >&2; exit 1; }
            files+=("$language_dir/$name")
        done
    done
}

# Print the finished output of an input file (classify_hplt.py compresses by default, and the
# launchers do not turn that off).
final_output_path() {
    local input="$1"
    echo "$output_root/$(basename "$(dirname "$input")")/$(basename "$input" .jsonl.zst).jsonl.zst"
}

# Fill the global array "pending" with the indices into "files" (the task numbers of
# classify_hplt.sh) of the files that have no finished output yet. Call list_hplt_files first.
list_pending_tasks() {
    pending=()
    local i
    for i in "${!files[@]}"; do
        [[ -f "$(final_output_path "${files[$i]}")" ]] || pending+=("$i")
    done
}

# Print the lock file classify_hplt.py holds while labelling an input file.
lock_path() {
    local input="$1"
    local language name
    language=$(basename "$(dirname "$input")")
    name=$(basename "$input" .jsonl.zst)
    echo "$output_root/$language/$name.lock"
}

# Remove an input's lock if the job that holds it is no longer running or pending (killed by
# SIGKILL or a node failure, so it could not remove the lock itself). A lock of this very job
# is also stale: each job labels a file at most once, so it can only be left by an earlier
# run of a requeued job. A lock is removed only when its job is known to have ended; locks of
# live jobs, and locks whose job's state cannot be read, are left alone, and classify_hplt.py
# then stops with an error for that file.
remove_stale_lock() {
    local lock owner state
    lock=$(lock_path "$1")
    [[ -f "$lock" ]] || return 0
    owner=$(sed -n 's/^slurm_job_id=//p' "$lock")
    if [[ -z "$owner" || "$owner" == "none" ]]; then
        echo "Lock $lock was not made by a SLURM job; leaving it (remove it by hand if no run is labelling the file)" >&2
        return 0
    fi
    if [[ "$owner" != "${SLURM_JOB_ID:-}" ]]; then
        # A job that has left the queue is unknown to squeue ("Invalid job id"); any other
        # squeue failure leaves the lock in place.
        if ! state=$(squeue -h -j "$owner" -o %T 2>&1); then
            if [[ "$state" != *"Invalid job id"* ]]; then
                echo "Cannot check job $owner, which holds $lock (squeue: $state); leaving the lock" >&2
                return 0
            fi
            state="GONE"
        fi
        case "$state" in
            ""|GONE|COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) ;;
            *)
                echo "Lock $lock is held by job $owner ($state); leaving it" >&2
                return 0
                ;;
        esac
    fi
    echo "Removing stale lock $lock (job $owner has ended)" >&2
    rm -f -- "$lock"
}

load_container_modules() {
    module purge
    module use /appl/local/laifs/modules
    module load Local-LAIF lumi-aif-singularity-bindings
    module list 2>&1
}
