#!/bin/bash
#SBATCH --job-name=label_horizon_teacher
#SBATCH --account=project_2020507
#SBATCH --partition=gpumedium
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72 # 72 CPUs per GPU
#SBATCH --gres=gpu:gh200:1
#SBATCH -o ../../logs/%x_%j.out
#SBATCH -e ../../logs/%x_%j.err

# Label a cross-labeller sample (sample_cross_labeller_eval.sh) with Horizon Labs' own labelling
# script, code/webtax/teacher_webtax.py of Horizon-Labs/multilingual-web-topic-format-base at
# the revision we evaluated, unchanged: WebOrganizer's prompt without URL or demonstrations,
# fixed label order, pages cut to 1,500 tokens, letter probabilities from the top 20 logprobs,
# topic and format in one run. Two labellers:
#   qwen:  Qwen/Qwen3.8-27B-FP8, their teacher (the default of their script)
#   gemma: google/gemma-4-31B-it quantised to FP8 by vLLM, their independent judge
# How to run on Roihu (submit from this directory so relative log paths resolve):
#   cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu
#   sbatch label_horizon_teacher.sh <qwen|gemma> <sample name>
# Pilot (every 48th page: 400 pages of the 19,200, all 128 languages; two timed chunks):
#   NSHARD=48 SHARD=0 CHUNK=200 sbatch label_horizon_teacher.sh qwen hplt-test-150
#   NSHARD=48 SHARD=0 CHUNK=200 sbatch label_horizon_teacher.sh gemma hplt-test-150
# Environment: NSHARD/SHARD split the sample (their script takes every NSHARD-th page from
# SHARD); SHARD defaults to the array task ID, so a full run can be an array, e.g.
#   NSHARD=4 sbatch --array=0-3 --time=04:00:00 label_horizon_teacher.sh qwen hplt-test-150
# 10,000 pages (hplt-test-major10-1000) take about 40 (qwen) and 50 (gemma) minutes plus model
# loading at the speeds of the hplt-test-150 runs (4.2 and 3.3 pages per second), so give 90 minutes:
#   sbatch --time=01:30:00 label_horizon_teacher.sh gemma hplt-test-major10-1000
# CHUNK (default 2000) is how many pages their script labels between saves. Their script
# doesn't resume: a rerun starts its shard again, so an existing output stops the job unless
# OVERWRITE=1.
#
# Their code is downloaded at the pinned revision into the Hugging Face cache, and the four
# files are checked against the SHA-256 sums of the versions we read before running them.
# Output: results/distribution/horizon-webtax/<labeller>/<sample name>/shard-<SHARD>-of-<NSHARD>.parquet
# (the sample's columns plus topic_<slug> and format_<slug>, 24 each; NaN where no answer
# letter was among the top 20 tokens). The .out log has timestamped progress lines for timing.

set -euo pipefail

usage="usage: sbatch label_horizon_teacher.sh <qwen|gemma> <sample name>"
labeller="${1:?$usage}"
sample_name="${2:?$usage}"
nshard="${NSHARD:-1}"
shard="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
chunk="${CHUNK:-2000}"
quant=""
case "$labeller" in
    qwen) model="Qwen/Qwen3.8-27B-FP8" ;;
    gemma) model="google/gemma-4-31B-it"; quant="fp8" ;;
    *) echo "$usage" >&2; exit 1 ;;
esac
if (( shard < 0 || shard >= nshard )); then
    echo "SHARD=$shard is out of range for NSHARD=$nshard" >&2
    exit 1
fi

base_dir="/scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer"
sample="$base_dir/results/statistics/cross_labeller/$sample_name/sample.parquet"
output_dir="$base_dir/results/distribution/horizon-webtax/$labeller/$sample_name"
output="$output_dir/shard-$shard-of-$nshard.parquet"
repo="Horizon-Labs/multilingual-web-topic-format-base"
revision="4ec2d09b46b89652ece638d9dee7865abcad6e69"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "labeller=$labeller model=$model quant=${quant:-none} sample=$sample shard=$shard/$nshard chunk=$chunk"
echo "output=$output"
if [[ ! -f "$sample" ]]; then
    echo "No sample at $sample; run sample_cross_labeller_eval.sh first" >&2
    exit 1
fi
if [[ -e "$output" && "${OVERWRITE:-0}" != 1 ]]; then
    echo "$output exists; set OVERWRITE=1 to label this shard again" >&2
    exit 1
fi
mkdir -p "$output_dir"

module purge
module load python-vllm/0.19.1
nvidia-smi --query-gpu=name,memory.total --format=csv

# Their code at the pinned revision, checked against the files we reviewed.
code_dir=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$repo', revision='$revision',
                        allow_patterns=['code/webtax/teacher_webtax.py', 'code/webtax/taxonomies/*.yaml',
                                        'code/sentiment/common.py']))")
(
    cd "$code_dir"
    sha256sum --check --strict <<'EOF'
2f02bdefb0e60010dbfc8593d604a385c0c052b201d2219fa4a8c8d15db6ca99  code/webtax/teacher_webtax.py
ac8ae2353c2b4569a0408692866a60a121716c838da7e968c90a43b614c127ec  code/sentiment/common.py
bbd8783bf3acbd8d506ef32c65507828c64246341810d6d2da9d70188c87f3ca  code/webtax/taxonomies/topics.yaml
b6357053453eeba9540235fbc8bf561d699f57a2a19c83ef49373d0ba69e5ba3  code/webtax/taxonomies/formats.yaml
EOF
)

export LABEL_IN="$sample" LABEL_OUT="$output" SHARD="$shard" NSHARD="$nshard" CHUNK="$chunk" MODEL="$model"
if [[ -n "$quant" ]]; then
    export QUANT="$quant"
else
    unset QUANT
fi
started=$(date +%s)
# Timestamp their progress lines ("labelled N of M") to time the chunks; vLLM's own log goes to .err.
python -u "$code_dir/code/webtax/teacher_webtax.py" | while IFS= read -r line; do
    printf '%s %s\n' "$(date '+%F %T')" "$line"
done
elapsed=$(( $(date +%s) - started ))

python - "$output" "$elapsed" <<'EOF'
import sys
import pandas as pd

path, elapsed = sys.argv[1], int(sys.argv[2])
frame = pd.read_parquet(path)
for prefix in ("topic_", "format_"):
    columns = [c for c in frame.columns if c.startswith(prefix)]
    if len(columns) != 24:
        raise SystemExit(f"{path}: {len(columns)} {prefix}* columns, expected 24")
    missing = int(frame[columns].isna().any(axis=1).sum())
    print(f"{prefix[:-1]}: {missing} of {len(frame)} pages without an answer letter in the top 20 tokens")
print(f"{len(frame)} pages in {elapsed} s including model loading: "
      f"{elapsed / 3600 * 200:.1f} BU at 200 BU per GPU-hour; {len(frame) / elapsed:.2f} pages per second overall")
EOF
echo "Done at $(date)"
