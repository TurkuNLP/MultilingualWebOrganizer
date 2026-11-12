#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/train_%j.out
#SBATCH -e ../../logs/train_%j.err

# Fine-tune mmBERT on the soft labels of a dataset made by create_datasets.sh.
# Usage (submit from this directory so relative log paths resolve):
#   sbatch train_classifier.sh <topic|format> <dataset_name> <run_name> [extra finetune.py args]
# e.g. sbatch train_classifier.sh topic hplt-v1 mmbert-mean
#      sbatch train_classifier.sh topic hplt-v1 mmbert-cls --classifier_pooling cls
# (mmBERT-base pools by mean by default; --classifier_pooling cls uses the first token.)
#      TRAIN_DOCUMENTS=1000000 sbatch --time=24:00:00 train_classifier.sh topic hplt-v1 mmbert-1m
#      TRAIN_DOCUMENTS=320 EVALUATIONS=2 WANDB= sbatch --partition=gputest --time=00:15:00 \
#          train_classifier.sh topic hplt-v1 smoke --max_eval_samples 500
#
# Training length: 1 epoch (one pass over the training split) by default; pass
# --num_train_epochs <n> for more. TRAIN_DOCUMENTS=<n> trains on n documents instead:
# n / 16 optimizer steps at the batch size below, or n / 64 with
# --gradient_accumulation_steps 8, so the amount of data stays the same when the batch
# size changes. One pass over the training split is its size (in the dataset's
# README.md); larger n repeats documents. Either way the run is evaluated and
# checkpointed EVALUATIONS times (default 20) at even intervals, the last at the final
# step, and the best evaluation's checkpoint is kept.
#
# Time: gpumedium allows at most 36 hours (sbatch --time=36:00:00). The October runs
# trained about 50,000 tokens (52 documents) per second, so one epoch of a dataset like
# hplt-random-v1 (7M training documents) takes about 30-37 hours, and 1M documents about
# 5 hours, plus evaluations. A run longer than one job resumes from its last checkpoint
# when the same command is submitted again, so keep checkpoints a few hours apart (the
# default 20 evaluations of a 36-hour epoch are about 1.8 hours apart) and chain jobs:
#   job=$(sbatch --parsable --time=36:00:00 train_classifier.sh topic hplt-random-v1 mmbert-1ep)
#   sbatch --dependency=afterany:$job --time=36:00:00 train_classifier.sh topic hplt-random-v1 mmbert-1ep
# If the first job finishes the epoch, cancel the second; it would only repeat the final
# evaluation and test predictions. Set TRAIN_DOCUMENTS or EVALUATIONS the same way for
# every job of a run (sbatch passes environment variables on to the job).
#
# Extra arguments override the defaults below (later options win), e.g.
# --learning_rate 5e-5 or --num_train_epochs 5. The model is written to
# results/models/<label type>/<dataset_name>/<run_name>; rerunning the same command
# resumes from the last checkpoint there.
#
# The input follows the dataset: the URL, a blank line, and the text (as in
# WebOrganizer's default classifiers, URLs percent-decoded) if it has a url column,
# the text alone if it was made with create_datasets.sh ... --no_url. Override with
# --template. The template is saved in the model's config.json.
#
# Weights & Biases: on by default (WANDB=online, project
# multilingual-weborganizer-finetune; both set below). Set them when submitting to
# change them for one run: WANDB=offline writes the run to <output dir>/wandb/ for
# `wandb sync` later, and an empty WANDB turns reporting off, e.g.
#   WANDB= sbatch train_classifier.sh topic hplt-v1 mmbert-mean
#   WANDB=offline WANDB_PROJECT=my-project TRAIN_DOCUMENTS=1000000 sbatch ... train_classifier.sh ...
# The W&B run is named <label type>/<dataset_name>/<run_name>, grouped by label type and
# dataset, and gets the training and evaluation curves, all finetune.py arguments, and
# the test metrics (overall, per language and per source). A resubmitted run continues
# the same W&B run (its ID is kept in <output dir>/wandb_run_id). Before the first
# online run, run `wandb login` once on a login node (it saves the key in ~/.netrc;
# never put the key in this script), and check that the module has wandb:
#   module load python-vllm/0.19.1 && python -c "import wandb"
# (if not: pip install --user wandb). If compute nodes can't reach wandb.ai, use
# WANDB=offline and sync from a login node: wandb sync <output dir>/wandb/offline-run-*

set -euo pipefail

usage="usage: sbatch train_classifier.sh <topic|format> <dataset_name> <run_name> [extra args]"
label_type="${1:?$usage}"
dataset_name="${2:?$usage}"
run_name="${3:?$usage}"
shift 3

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"

module purge
module load python-vllm/0.19.1

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
python_script="$base_dir/scripts/model_training/finetune.py"
dataset_dir="$base_dir/results/datasets/$label_type/$dataset_name"
output_dir="$base_dir/results/models/$label_type/$dataset_name/$run_name"
echo "Dataset: $dataset_dir"
echo "Output:  $output_dir"
nvidia-smi --query-gpu=name,memory.total --format=csv

# W&B reporting defaults; a value set when submitting wins, and WANDB= (empty) turns
# reporting off. (${WANDB-...} keeps an empty value, ${WANDB:-...} would replace it.)
WANDB="${WANDB-online}"
WANDB_PROJECT="${WANDB_PROJECT:-multilingual-weborganizer-finetune}"

# Reporting: none unless WANDB is online or offline.
report_args=(--report_to none)
case "$WANDB" in
    "") echo "W&B: off" ;;
    online|offline)
        if ! python -c "import wandb" 2>/dev/null; then
            echo "WANDB=$WANDB, but the python-vllm module has no wandb; install it with pip install --user wandb," \
                 "or submit with WANDB= to train without reporting" >&2
            exit 1
        fi
        if [[ "$WANDB" == online && -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
            echo "WANDB=online, but no W&B key is set up; run wandb login on a login node first," \
                 "or submit with WANDB=offline or WANDB=" >&2
            exit 1
        fi
        export WANDB_MODE="$WANDB"
        export WANDB_PROJECT
        export WANDB_RUN_GROUP="$label_type/$dataset_name"
        export WANDB_TAGS="$label_type,$dataset_name"
        # Run files next to the model, caches in scratch rather than the small home directory.
        export WANDB_DIR="$output_dir"
        export WANDB_CACHE_DIR="$base_dir/cache/wandb"
        export WANDB_DATA_DIR="$base_dir/cache/wandb-data"
        mkdir -p "$output_dir" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
        # One W&B run per output directory, so a resubmitted (resumed) job continues it.
        run_id_file="$output_dir/wandb_run_id"
        if [[ ! -s "$run_id_file" ]]; then
            python -c "import wandb.util; print(wandb.util.generate_id())" > "$run_id_file"
        fi
        export WANDB_RUN_ID="$(<"$run_id_file")"
        export WANDB_RESUME=allow
        report_args=(--report_to wandb --run_name "$label_type/$dataset_name/$run_name")
        echo "W&B: $WANDB_MODE, project $WANDB_PROJECT, run $label_type/$dataset_name/$run_name (ID $WANDB_RUN_ID)"
        ;;
    *)
        echo "WANDB must be online, offline or empty (no reporting), got '$WANDB'" >&2
        exit 1
        ;;
esac

if [[ -n "${MAX_STEPS:-}${EVAL_STEPS:-}" ]]; then
    echo "MAX_STEPS and EVAL_STEPS were replaced by TRAIN_DOCUMENTS and EVALUATIONS" >&2
    exit 1
fi
evaluations="${EVALUATIONS:-20}"
# At least 2: an interval of 1 would mean every step instead of the whole run.
if ! [[ "$evaluations" =~ ^[1-9][0-9]*$ ]] || (( evaluations < 2 )); then
    echo "EVALUATIONS must be an integer of at least 2, got '$evaluations'" >&2
    exit 1
fi
if [[ -n "${TRAIN_DOCUMENTS:-}" ]] && ! [[ "$TRAIN_DOCUMENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_DOCUMENTS must be a positive integer, got '$TRAIN_DOCUMENTS'" >&2
    exit 1
fi
if [[ -n "${TRAIN_DOCUMENTS:-}" ]]; then
    length_args=(--max_train_documents "$TRAIN_DOCUMENTS")
else
    length_args=(--num_train_epochs 1)
fi
# The interval is a fraction of the run, because finetune.py derives the number of
# steps from the batch size. The Trainer turns it into steps and also evaluates the
# final step.
interval=$(awk -v n="$evaluations" 'BEGIN { printf "%.6g", 1 / n }')
# Checkpoints at every evaluation, as --load_best_model_at_end requires.
schedule_args=("${length_args[@]}" --eval_strategy steps --eval_steps "$interval"
               --save_strategy steps --save_steps "$interval")
echo "Schedule: ${schedule_args[*]}"

# Defaults: the best settings of the 50,000-document hyperparameter searches (trial 13 of
# hp_search/docs50k, best for both topic and format): learning rate 3.05e-5, effective
# batch size 16 (8 x 2 accumulation steps), weight decay 0.088, warmup 7.6% of the steps,
# mean pooling. Extra arguments override them. Batches are grouped by length,
# so the longest batches hold 8 documents of up to 8,192 tokens; check gpu_mem_free
# in the first log lines and lower the per-device batch size if memory runs out.
python "$python_script" \
    --model_name jhu-clsp/mmBERT-base \
    --dataset_path "$dataset_dir" \
    --output_dir "$output_dir" \
    --do_train \
    --do_eval \
    --do_predict \
    --bf16 \
    --max_length 8192 \
    --decode_url true \
    --learning_rate 3.05e-5 \
    --weight_decay 0.088284 \
    --warmup_steps 0.076468 \
    --classifier_pooling mean \
    "${schedule_args[@]}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --load_best_model_at_end \
    --metric_for_best_model js_divergence_bits \
    --greater_is_better false \
    --logging_steps 50 \
    "${report_args[@]}" \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 16 \
    --seed 42 \
    "$@"
