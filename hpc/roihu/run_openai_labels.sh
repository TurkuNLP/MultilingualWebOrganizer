#!/bin/bash
# Label the parallel Nemotron-CC documents with an OpenAI model through the API:
# one hard label (plus the model's explanation) per document, using exactly the
# prompts of the open-model runs. Not a SLURM job: it needs no GPU and runs on a
# Roihu-GPU login node (the python-vllm module is built for the GPU nodes' ARM
# CPUs, so it doesn't work on the x86 Roihu-CPU login nodes).
#
# Usage (run inside tmux or screen, so a dropped SSH connection doesn't stop it):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   bash run_openai_labels.sh <model> [max_documents] [reasoning_effort]
#
#   bash run_openai_labels.sh gpt-6-luna 20 none     # pilot: 20 documents per run
#   bash run_openai_labels.sh gpt-6-luna 2000 none   # full run; resumes the pilot files
#
# max_documents defaults to 2000 per language and label type. reasoning_effort is one
# of none, minimal, low, medium, high; omit it to use the model's default (more
# reasoning tokens, billed as output). Use the same effort for the pilot and the full
# run, so both parts of each file are comparable.
#
# API key: export OPENAI_API_KEY, or store the key in a file readable only by you:
#   mkdir -p ~/.config/openai && (umask 077; cat > ~/.config/openai/api_key)
# (OPENAI_API_KEY_FILE overrides that path.) The key is never printed or logged.
#
# Optional environment overrides:
#   LABEL_TYPES="topic format"   label types to run
#   LANGUAGES="english finnish"  subset of languages (default: all six)
#   CONCURRENCY=4                parallel API requests per run
#   ASSUME_YES=1                 skip the confirmation prompt
#
# Each language uses the same seed and dataset split as array_generate_distribution.sh
# (English seed 0, Finnish 1, ...), so prompts match the open-model runs document by
# document. Output: results/api/<model>/<label type>/<language>-parallel.jsonl
# Logs: logs/openai_<model>_<label type>_<language>_<timestamp>.log
# Runs are sequential and stop at the first failure; rerun the same command to resume.

module purge
module load python-vllm/0.19.1

set -euo pipefail

usage="usage: bash run_openai_labels.sh <model> [max_documents] [reasoning_effort]"
model="${1:?$usage}"
max_documents="${2:-2000}"
reasoning_effort="${3:-}"
label_types="${LABEL_TYPES:-topic format}"
concurrency="${CONCURRENCY:-4}"

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/data_processing/generate_openai_labels.py"

# Same languages, dataset configs, and splits as array_generate_distribution.sh.
languages=(english finnish german hungarian icelandic ukrainian)
dataset_configs=(eng_Latn fin_Latn deu_Latn hun_Latn isl_Latn ukr_Cyrl)
tower72b_languages=(finnish german)
selected_languages="${LANGUAGES:-${languages[*]}}"

for language in $selected_languages; do
    if [[ " ${languages[*]} " != *" $language "* ]]; then
        echo "Unknown language: $language (expected some of: ${languages[*]})" >&2
        exit 1
    fi
done
for label_type in $label_types; do
    if [[ "$label_type" != "topic" && "$label_type" != "format" ]]; then
        echo "Unknown label type: $label_type (expected topic and/or format)" >&2
        exit 1
    fi
done

# Read the API key from the environment or from a private key file. Never echo it.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    key_file="${OPENAI_API_KEY_FILE:-$HOME/.config/openai/api_key}"
    if [[ ! -r "$key_file" ]]; then
        echo "Set OPENAI_API_KEY or store the key in $key_file (see the header of this script)" >&2
        exit 1
    fi
    key_permissions="$(stat -c %a "$key_file")"
    if [[ "$key_permissions" != "600" && "$key_permissions" != "400" ]]; then
        echo "$key_file must be readable only by you: chmod 600 $key_file" >&2
        exit 1
    fi
    OPENAI_API_KEY="$(tr -d '[:space:]' < "$key_file")"
    export OPENAI_API_KEY
fi

reasoning_args=()
if [[ -n "$reasoning_effort" ]]; then
    reasoning_args=(--reasoning-effort "$reasoning_effort")
fi

run_count=$(( $(wc -w <<< "$label_types") * $(wc -w <<< "$selected_languages") ))
echo "Model:              $model"
echo "Reasoning effort:   ${reasoning_effort:-model default}"
echo "Label types:        $label_types"
echo "Languages:          $selected_languages"
echo "Documents per run:  $max_documents ($run_count runs, up to $(( run_count * max_documents )) API requests)"
echo "Concurrency:        $concurrency"
echo "Output:             $base_dir/results/api/$model/"
if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    read -r -p "This calls a paid API. Continue? [y/N] " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

mkdir -p "$base_dir/logs"
for label_type in $label_types; do
    for index in "${!languages[@]}"; do
        language="${languages[$index]}"
        if [[ " $selected_languages " != *" $language "* ]]; then
            continue
        fi
        if [[ "$language" == "english" ]]; then
            dataset_split=parallel
        elif [[ " ${tower72b_languages[*]} " == *" $language "* ]]; then
            dataset_split=tower72b_parallel
        else
            dataset_split=tower9b_parallel
        fi
        output="$base_dir/results/api/$model/$label_type/${language}-parallel.jsonl"
        log="$base_dir/logs/openai_${model}_${label_type}_${language}_$(date +%Y%m%d-%H%M%S).log"
        echo "=== $(date '+%F %T') $label_type / $language -> ${output#"$base_dir"/} (log: ${log#"$base_dir"/})"
        python "$python_script" \
            --input-dataset MultiSynt/MT-Nemotron-CC \
            --dataset-config "${dataset_configs[$index]}" \
            --dataset-split "$dataset_split" \
            --language "$language" \
            --seed "$index" \
            --label-type "$label_type" \
            --model "$model" \
            --max-documents "$max_documents" \
            --concurrency "$concurrency" \
            ${reasoning_args[@]+"${reasoning_args[@]}"} \
            --output "$output" \
            2>&1 | tee "$log"
    done
done
echo "=== $(date '+%F %T') all runs finished"
