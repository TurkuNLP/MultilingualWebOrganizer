#!/bin/bash
# Submit hplt-sampler runs (https://github.com/TurkuNLP/hplt-sampler) for every language
# code of a languages file, a few at a time. Run it with bash on a login node, not sbatch:
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   bash submit_hplt_sampler.sh <clean|noisy|poor> [parallel_runs] [languages_file]
# e.g. bash submit_hplt_sampler.sh clean 4
#
# The languages come from data/qwen_label_languages.txt (128 languages, one code each)
# unless another file is given. Each language code is one hplt-sampler run of 50,000
# documents with a fixed seed (all documents of a code that has fewer). A language with
# several codes in another file gets one run per code; collect_hplt_samples.sh combines
# them in proportion to their document counts.
# Codes without a directory in the collection are skipped with a warning, and so are codes
# whose run has finished (samples.jsonl written, per-shard samples removed), so the script
# can be run again after cancelled or failed runs. An unfinished run resumes: the sampler
# keeps its cached counts and finished per-shard samples, and the fixed seed makes the rest
# match them. Check with squeue --me that none of the runs is still queued or running first.
#
# At most parallel_runs (default 4) orchestrators run at once: each run waits for the one
# submitted parallel_runs places before it (afterany, so a failed run does not block the
# rest). The runs share your queue limits, so set BATCH in hplt-sampler.conf to about
# 180 / parallel_runs on LUMI small.
#
# Needs, in the sampler clone: hplt-sampler.conf with ACCOUNT set, and the fix that keeps
# the collections apart (branch fix-collection-clash) before sampling noisy or poor.
# Results: <sampler>/results-v4-<code>-all/ (clean) or results-v4-<collection>-<code>-all/;
# the submitted job IDs are listed in <sampler>/submitted-<collection>-<date>.tsv.

set -euo pipefail

usage="usage: bash submit_hplt_sampler.sh <clean|noisy|poor> [parallel_runs] [languages_file]"
collection="${1:?$usage}"
parallel_runs="${2:-4}"
case "$collection" in
    clean|noisy|poor) ;;
    *) echo "$usage" >&2; exit 1 ;;
esac
if ! [[ "$parallel_runs" =~ ^[1-9][0-9]*$ ]]; then
    echo "parallel_runs must be a positive integer; $usage" >&2
    exit 1
fi

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
sampler_dir="/scratch/project_462001516/users/tarkkaot/hplt-sampler"
languages_file="${3:-$base_dir/data/qwen_label_languages.txt}"
local_root="/appl/local/openeurollm/training/catalogue/hplt/4.0/$collection"
n_samples=50000
seed=42

if [[ ! -f "$sampler_dir/hplt-sampler.conf" ]]; then
    echo "No $sampler_dir/hplt-sampler.conf; copy presets/lumi.conf there and set ACCOUNT" >&2
    exit 1
fi
if [[ "$collection" != clean ]] && ! grep -qF 'basename "${LOCAL_ROOT%/}"' "$sampler_dir/sl-orchestrator"; then
    echo "The sampler in $sampler_dir lacks the clean/noisy fix; $collection runs would reuse clean counts" >&2
    exit 1
fi

# The run directory, as the sampler names it: the default (clean) collection has no tag.
run_tag="v4"
[[ "$collection" != clean ]] && run_tag="v4-$collection"

# The language codes that have a directory in the collection and no finished run. A run
# counts as finished where the sampler wrote it or after moving to results/<collection>/.
# An unfinished run in results/<collection>/ would not resume there, so it stops the script.
codes=()
names=()
moved_unfinished=()
while IFS=: read -r name code_list || [[ -n "$name" ]]; do
    [[ -z "${name// }" ]] && continue
    for code in $code_list; do
        run_name="results-$run_tag-$code-all"
        finished=""
        for run_dir in "$sampler_dir/$run_name" "$sampler_dir/results/$collection/$run_name"; do
            if [[ -f "$run_dir/samples.jsonl" && ! -d "$run_dir/samples" ]]; then
                finished="$run_dir"
            fi
        done
        if [[ ! -d "$local_root/$code" ]]; then
            echo "Skipping $name: no directory $local_root/$code" >&2
        elif [[ -n "$finished" ]]; then
            echo "Skipping $name ($code): finished in $finished" >&2
        elif [[ -d "$sampler_dir/results/$collection/$run_name" ]]; then
            moved_unfinished+=("$sampler_dir/results/$collection/$run_name")
        else
            codes+=("$code")
            names+=("$name")
        fi
    done
done < "$languages_file"
if (( ${#moved_unfinished[@]} > 0 )); then
    echo "These unfinished runs were moved out of $sampler_dir, where the sampler resumes them." >&2
    echo "Move them back first (mv <run> $sampler_dir/):" >&2
    printf '  %s\n' "${moved_unfinished[@]}" >&2
    exit 1
fi

log="$sampler_dir/submitted-$collection-$(date +%Y%m%d-%H%M%S).tsv"
printf 'language\tcode\tjob_id\tafter\n' > "$log"
cd "$sampler_dir"
job_ids=()
for i in "${!codes[@]}"; do
    dependency=()
    after=""
    if (( i >= parallel_runs )); then
        after="${job_ids[$((i - parallel_runs))]}"
        dependency=(--dependency="afterany:$after")
    fi
    job_id=$(sbatch --parsable --account=project_462001516 --partition=small --mem=2G \
        ${dependency[@]+"${dependency[@]}"} \
        sl-orchestrator "${codes[$i]}" --corpus v4 --local-root "$local_root" \
        --n-samples "$n_samples" --seed "$seed")
    job_ids+=("$job_id")
    printf '%s\t%s\t%s\t%s\n' "${names[$i]}" "${codes[$i]}" "$job_id" "$after" >> "$log"
    echo "${names[$i]} (${codes[$i]}): job $job_id${after:+, after $after}"
done
echo "Submitted ${#job_ids[@]} runs; job IDs in $log"
