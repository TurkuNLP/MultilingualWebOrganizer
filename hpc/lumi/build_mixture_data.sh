#!/bin/bash
#SBATCH --job-name=build_mixture_data
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH -o ../../logs/%x_%A_%a.out
#SBATCH -e ../../logs/%x_%A_%a.err

# Build the Megatron training data of a mixture: one array task per classifier output file
# of the language. Each task
#   1. selects the file's documents with the mixture's selection probabilities
#      (scripts/data_processing/select_mixture_documents.py) into a plain JSONL file,
#   2. tokenizes them as the base models' data was tokenized (preprocess_oellm.sh: Megatron's
#      preprocess_data.py, openeurollm/tokenizer-256k, --append-eod),
#   3. checks that the tokenized file has one document per selected document, and deletes
#      the plain JSONL file.
# The mixture comes from scripts/data_processing/compute_mixture.py:
#   results/mixtures/hplt-4.0-clean/<language>/<mixture>.json
#
# How to run on LUMI (submit from this directory so relative log paths resolve):
#   cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi
#   bash build_mixture_data.sh list <language> [mixture]     # array index -> file
#   sbatch --array=0-155%30 build_mixture_data.sh <language> [mixture]
# mixture defaults to "both". %30 limits the plain JSONL files on disk at a time (a few GB
# each). A task whose output is complete exits at once, so a failed or timed-out task can
# be submitted again with the same command.
#
# Output (DATA_PATH for training is the megatron directory):
#   data/megatron/oellm-v1-256k/hplt-4.0-clean-<mixture>/<language>/<bucket>_<part>_text_document.{bin,idx}
#   data/megatron/oellm-v1-256k/hplt-4.0-clean-<mixture>/selection/<language>/<bucket>_<part>.{selection.json,labels.tsv}

set -euo pipefail

base_dir="/scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer"
SIF="$base_dir/environments/lumi/vllm.sif"
select_script="$base_dir/scripts/data_processing/select_mixture_documents.py"
preprocess_script="/flash/project_465002530/tools/OpenEuroLLM-NVIDIA-Megatron-LM/tools/preprocess_data.py"
tokenizer="openeurollm/tokenizer-256k"
seed=42

if [[ "${1:-}" == "list" ]]; then
    mode=list
    shift
else
    mode=run
fi
language="${1:?usage: sbatch --array=... build_mixture_data.sh <language> [mixture] | bash build_mixture_data.sh list <language> [mixture]}"
mixture_name="${2:-both}"

prediction_dir="$base_dir/results/classifier_predictions/mmbert-1/hplt-4.0-clean/$language"
mixture="$base_dir/results/mixtures/hplt-4.0-clean/$language/$mixture_name.json"
output_root="$base_dir/data/megatron/oellm-v1-256k/hplt-4.0-clean-$mixture_name"
megatron_dir="$output_root/$language"
selection_dir="$output_root/selection/$language"
work_dir="$output_root/work/$language"

# The classifier output files in bucket, part order; the array index is the line number.
mapfile -t files < <(find "$prediction_dir" -maxdepth 1 -name '*_*.jsonl.zst' -printf '%f\n' | sort -t_ -k1,1n -k2,2n)
if (( ${#files[@]} == 0 )); then
    echo "No classifier output in $prediction_dir" >&2
    exit 1
fi
if [[ -n "$(find "$prediction_dir" -maxdepth 1 \( -name '*.partial' -o -name '*.lock' \) -print -quit)" ]]; then
    echo "$prediction_dir has unfinished files (.partial or .lock)" >&2
    exit 1
fi

if [[ "$mode" == list ]]; then
    for i in "${!files[@]}"; do
        echo "$i ${files[$i]}"
    done
    echo "${#files[@]} files: sbatch --array=0-$(( ${#files[@]} - 1 ))%30 build_mixture_data.sh $language $mixture_name" >&2
    exit 0
fi

task="${SLURM_ARRAY_TASK_ID:?submit as an array: sbatch --array=0-$(( ${#files[@]} - 1 ))%30 build_mixture_data.sh $language}"
if (( task >= ${#files[@]} )); then
    echo "Array index $task is out of range for ${#files[@]} files" >&2
    exit 1
fi
file="${files[$task]}"
name="${file%.jsonl.zst}"
if [[ ! -f "$mixture" ]]; then
    echo "No mixture $mixture; run compute_mixture.py first" >&2
    exit 1
fi

jsonl="$work_dir/$name.jsonl"
summary="$selection_dir/$name.selection.json"
labels="$selection_dir/$name.labels.tsv"
prefix="$megatron_dir/$name"
idx="${prefix}_text_document.idx"
mkdir -p "$megatron_dir" "$selection_dir" "$work_dir"

echo "Job ${SLURM_ARRAY_JOB_ID}_${task} on $(hostname) at $(date)"
echo "Input:   $prediction_dir/$file"
echo "Mixture: $mixture"
echo "Output:  ${prefix}_text_document.{bin,idx}"

# Selected documents according to the summary; a check of the tokenized file against it.
selected_documents() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_documents"])' "$summary"
}
# Sequences in a Megatron .idx file (header: 9-byte magic, version, dtype, sequence count).
idx_sequences() {
    python3 - "$1" <<'EOF'
import struct, sys
with open(sys.argv[1], "rb") as fh:
    header = fh.read(9 + 8 + 1 + 8)
if header[:9] != b"MMIDIDX\x00\x00":
    sys.exit(f"{sys.argv[1]}: not a Megatron index file")
print(struct.unpack("<Q", header[18:26])[0])
EOF
}

if [[ -f "$idx" && -f "$summary" ]]; then
    if [[ "$(idx_sequences "$idx")" == "$(selected_documents)" ]]; then
        echo "Already complete: $idx"
        exit 0
    fi
    echo "$idx does not match $summary; building it again" >&2
    rm -f "$idx" "${prefix}_text_document.bin"
fi

# 1. Selection, unless an earlier run of this task completed it and kept the JSONL file.
if [[ ! -f "$summary" || ! -f "$jsonl" ]]; then
    module purge
    module use /appl/local/laifs/modules
    module load Local-LAIF lumi-aif-singularity-bindings
    srun --ntasks=1 singularity run -B /scratch/project_462001516 "$SIF" python "$select_script" \
        --mixture "$mixture" \
        --input "$prediction_dir/$file" \
        --output "$jsonl" \
        --labels "$labels" \
        --summary "$summary" \
        --seed "$seed"
fi

# 2. Tokenization, as in preprocess_oellm.sh.
module purge
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
module use /appl/local/csc/modulefiles
module load pytorch/2.5
export HF_HOME="/scratch/project_462001516/cache/huggingface"
if [[ -f ~/.huggingface/token ]]; then
    HUGGING_FACE_HUB_TOKEN="$(cat ~/.huggingface/token)"
    export HUGGING_FACE_HUB_TOKEN
fi

srun --ntasks=1 python3 "$preprocess_script" \
    --input "$jsonl" \
    --output "$prefix" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$tokenizer" \
    --append-eod \
    --log-interval 100000 \
    --workers "${SLURM_CPUS_PER_TASK:-32}"

# 3. Check and clean up.
sequences="$(idx_sequences "$idx")"
expected="$(selected_documents)"
if [[ "$sequences" != "$expected" ]]; then
    echo "$idx has $sequences documents, but $expected were selected; keeping $jsonl" >&2
    exit 1
fi
rm -f "$jsonl"
echo "Done at $(date): $expected documents in $idx"
