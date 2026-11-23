# Multilingual WebOrganizer

Code for training multilingual versions of the [WebOrganizer](https://github.com/CodeCreator/WebOrganizer) topic and format classifiers. An open LLM labels web pages in many European languages with a **probability distribution** over 26 topic and 26 format labels. A multilingual encoder (mmBERT) is then trained on these soft labels, so that large web corpora can be organised by topic and format in every language.

The code runs on the CSC supercomputers Roihu and LUMI, through SLURM.

## Status (23 November 2026)

**Now: labelling HPLT 4.0 with the trained classifiers.** The two URL models (`mmbert-topic-1`, `mmbert-format-1`) label every document of HPLT 4.0's clean collection in German, Spanish, Finnish and Hungarian, on LUMI (`classify_hplt.py`; see [Labelling corpora with the classifiers](#labelling-corpora-with-the-classifiers)). Results: `results/classifier_predictions/mmbert-1/hplt-4.0-clean/<language>/`.

- **German is complete:** 156 files, about 1.05 billion documents.
- **Spanish** has 159 of 160 files, **Finnish** 7 of 10 and **Hungarian** 9 of 15. The unfinished files are resubmitted on small-g and continue from their partial output.
- **Speed:** about 70,000 tokens per second per MI250X GCD for both models together. Documents in HPLT quality buckets 9 and 10 are 2–4 times longer (2,100–3,800 tokens) than in buckets 5–8 (550–1,100), so those files take longest. LUMI GPU jobs currently wait 2–4 days in the queue: set time limits per file from these rates, with margin, so that no job has to be resubmitted.
- **Size:** about 1.3 KB per document compressed, roughly 2.9 TB for the four languages.

**Next:**

- **Label proportions of German** (`sbatch label_proportions.sh deu_Latn`): the top-label shares of topics, formats and topic × format pairs, by documents and by characters, in two variants: all labels, and *Bad example* dropped with *Other* fixed at its natural share.
- **Monolingual pretraining with WebOrganizer's optimised mixture** (Wettig et al., 2025, Table 9, column *Both*, i.e. tuned for MMLU and HellaSwag), applied as a topic × format mixture: the product of the topic and format mixtures, each pair capped at the data it has, the rest scaled up (the paper's Section 4.2). Training uses `hpc/lumi/train-0.4B-ne.sh` (0.4B parameters, 100B tokens); base models on the natural distribution are already trained. German and Spanish come first. Estimates from the random samples: each label's target is reachable within about one epoch, but capping the rare pairs leaves *Science & Tech.* at 12–15 % instead of 25 %. Finnish and Hungarian are too small for the mixture at 100B tokens: *Science & Tech.* alone would need 11–19 epochs.
- **Human check of the URL effect:** Qwen3.8-27B labelled with the URL in its prompt, so part of the URL models' better agreement with it may be agreement on URL-driven decisions. The human annotations (`data/human_evaluation/`) can test this.

**Done:**

- **Labelling model: Qwen3.8-27B,** chosen over four other open models, a decision model built on it (Clef) and two OpenAI models, and checked against a human annotator (`docs/labelling_model_recommendation.md`).
- **Training data:** Qwen3.8-27B label distributions of random samples of HPLT 4.0's clean, noisy and poor collections in 128 languages (`data/qwen_label_languages.txt`). The datasets `results/datasets/{topic,format}/hplt-finetune` (URL and text) and `hplt-finetune-nourl` (text only) hold 7,288,056 documents: up to 25,000 clean, 25,000 noisy and 10,000 poor documents per language, split 94 / 1 / 5 % per language. 0.3 % of the training documents are near duplicates; they barely affect the evaluation (`find_duplicates.sh`).
- **Classifiers:** four mmBERT-base models (topic and format, each with and without the URL), one epoch each: `results/models/<label type>/hplt-finetune/mmbert-{topic,format}[-nourl]-1`. On the test split, without the test documents that duplicate training documents (`notebooks/classifier_final.ipynb`):
  - Agreement with Qwen3.8-27B's top label is 88.7 % (topic) and 88.9 % (format) with the URL, against 86.1 % and 85.7 % without. The URL model is clearly better in 125 of the 128 languages for each task and worse in none; it gains most on poor-quality pages.
  - On WebOrganizer's measure (documents whose labeller's top label is above 75 %, the 24 shared labels), the URL models are above WebOrganizer's reported English classifiers: label average 94.7 % vs 93.5 % (topic) and 96.5 % vs 91.8 % (format). The setups differ (labeller, languages, data), so this is not like-for-like.
  - The models were still improving at the end of the epoch.
- **First classifiers (October 2026),** on `hplt-v1` (5,000 documents for each of 36 European languages): test accuracy 0.860 (topic) and 0.847 (format).

## How it works

1. **Sample documents.** `scripts/data_processing/sample_hplt_by_quality.py` draws a sample per language from one of HPLT 4.0's collections (clean, noisy or poor), stratified by HPLT's quality bucket. The languages come from a file of `Language: code [code ...]` lines. The sample is split evenly over a language's codes, then its quality buckets, then the sources within a bucket. By default the documents are taken from the start of each file; with `--scan-factor N` they are drawn at random from the first N times as many as needed (`0`: from the whole source). The files hold their documents grouped by crawl, so the start of a file is not representative. `--method bernoulli` therefore instead keeps every document of every file with the same probability, with no quotas, so the sample follows the composition of the full data. A document whose ID is already in the sample is skipped, and the output is shuffled. It runs on LUMI via `hpc/lumi/sample_hplt_by_quality.sh`. The Roihu copy is in `data/hplt4/<language>.jsonl`: 5,000 documents for each of 36 languages. The current training data uses random samples made with [hplt-sampler](https://github.com/TurkuNLP/hplt-sampler) instead (see [Running on LUMI](#running-on-lumi)).
2. **Label them with label distributions.** `scripts/data_processing/generate_label_distributions.py` shows the model the label list under single-letter IDs, with worked examples, and reads the probabilities of the answer letters in one forward pass (vLLM). Each document gets a probability for every label, normalised the same way as in WebOrganizer.
3. **Build training data.** `scripts/model_training/create_datasets.py` turns the label files into train, validation and test splits. It splits by document ID, so the translations of a document never end up in different splits. With `--stratify_by_language`, every language is split in exactly the requested ratios instead of approximately, which matters for small languages.
4. **Train the classifier.** `scripts/model_training/finetune.py` fine-tunes mmBERT (`jhu-clsp/mmBERT-base`) with cross-entropy against the label distributions. As in WebOrganizer's default classifiers, the input is the URL, a blank line, and the text; URLs are percent-decoded first. A dataset made with `create_datasets.py --no_url` has no URLs and gives a text-only classifier. `--template` overrides the input, and the template is saved in the model's `config.json`. `hp_search.py` and `summarize_hp_search.py` run a random hyperparameter search, one trial per SLURM array task.
5. **Evaluate.** `scripts/data_analysis/`, `scripts/model_evaluation/` and `notebooks/` compare models with each other, across languages, with WebOrganizer's labels and classifiers, and with human labels.
6. **Label corpora.** `scripts/data_processing/classify_hplt.py` runs the trained topic and format classifiers over HPLT 4.0 files, both on each document in one pass, with exactly the input of training. `scripts/data_analysis/label_proportions.py` then measures the label composition of a labelled language.

### Labels

The label definitions are in `scripts/data_processing/topics.yaml` and `formats.yaml`, and the worked examples in `topic_examples.yaml` and `format_examples.yaml`. Each set has WebOrganizer's 24 labels plus two fallbacks:

- **Other:** the page is clearly outside all other labels.
- **Bad example:** the content is unintelligible or too short to label.

### Prompt order

To keep the model from favouring answer positions, the label and example order changes per document. With `--prompt-order rotated`, documents share one of 26 rotations, so vLLM can reuse the cached start of the prompt; this cuts labelling cost substantially. Each language uses its own seed, so a position preference can't make translations look more consistent than they are.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/data_processing/` | Sampling, labelling (`generate_label_distributions.py`, plus `generate_label_clef.py` for Clef and `generate_openai_labels.py` for OpenAI models), prompts, labels, the human-evaluation sample, labelling corpora with the trained classifiers (`classify_hplt.py`), and shared code (`labeling_common.py`, `distribution_records.py`). `generate_multilabel.py` is the earlier hard-label pipeline |
| `scripts/model_training/` | Dataset creation, classifier fine-tuning, hyperparameter search, token counts |
| `scripts/model_evaluation/` | Predictions of WebOrganizer's original classifiers on our datasets (`predict_weborganizer.py`) |
| `scripts/data_analysis/` | Evaluation: consistency across languages, agreement with WebOrganizer, shared metrics (`distribution_metrics.py`), duplicates, label proportions of labelled corpora (`label_proportions.py`) |
| `hpc/roihu/`, `hpc/lumi/` | SLURM launch scripts; each documents its usage in its header. `hpc/lumi/train-0.4B-ne.sh` pretrains a 0.4B-parameter model with Megatron-LM (billed to `project_465002530`) |
| `configs/` | YAML configs per system: `distribution/` (labelling; `lumi/distribution/generate_hplt.yaml` for the HPLT samples), `classifier/` (search space), `multilabel/` (the earlier hard-label pipeline) |
| `environments/lumi/` | `vllm.sif`, a link to the LUMI AI Factory container the LUMI scripts run in |
| `notebooks/` | Analyses, saved with their outputs |
| `docs/` | Reports and annotation guidelines |
| `data/`, `results/`, `logs/` | Inputs, outputs and job logs (not version-controlled). Language lists: `data/oellm_languages.txt` (the 38 OpenEuroLLM languages), `data/hplt_languages.txt` (all 187 HPLT 4.0 languages) and `data/qwen_label_languages.txt` (the 128 labelled languages, with their codes) |

## Running on Roihu

Scripts load the `python-vllm/0.19.1` module, which provides Python 3.12, vLLM 0.19.1, torch and transformers 5.8 and runs only on the (ARM) GPU nodes. CPU-only jobs (`find_duplicates.sh`) load `python-data` and run on the (x86) CPU nodes; submit them from a Roihu-CPU login node. Submit from `hpc/roihu/` so the relative log paths resolve; logs go to `logs/`.

```bash
cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu

# Label the HPLT samples: 72 tasks, 36 languages x {topic, format}
sbatch array_generate_distribution_hplt.sh [max_documents]

# Label the parallel MT-Nemotron-CC evaluation documents (6 languages x 2 label types)
sbatch array_generate_distribution.sh Qwen/Qwen3.8-27B 2000

# Build a dataset from a labelling run, then train (or search hyperparameters)
sbatch create_datasets.sh Qwen3.8-27B-hplt-annotations topic hplt-v1
# or, from the clean, noisy and poor random results copied from LUMI (25,000 / 25,000 / 10,000 per language)
sbatch create_hplt_dataset.sh topic hplt-finetune
sbatch create_hplt_dataset.sh topic hplt-finetune-nourl --no_url
sbatch train_classifier.sh topic hplt-finetune mmbert-topic-1
sbatch count_language_tokens.sh hplt-random           # documents and mmBERT tokens per language
sbatch find_duplicates.sh topic/hplt-finetune         # exact and near duplicates within and across splits
sbatch train_classifier.sh topic hplt-v1 mmbert
WANDB= sbatch train_classifier.sh topic hplt-v1 mmbert   # without Weights & Biases reporting
sbatch --array=0-15%4 hp_search_classifier.sh topic hplt-v1 search1

# Compare with the original WebOrganizer: its annotations as our datasets, its classifiers on our data
sbatch create_weborganizer_eval_dataset.sh
sbatch predict_weborganizer.sh topic hplt-finetune test
```

`train_classifier.sh` reports to Weights & Biases by default (project `multilingual-weborganizer-finetune`); run `wandb login` once on a login node first. Submit with `WANDB=offline` to sync later with `wandb sync`, or `WANDB=` to turn reporting off. `gpumedium` jobs run for at most 36 hours, and a run resumes from its last checkpoint when resubmitted, so long runs are chained jobs. By default a run is one epoch, evaluated and checkpointed 20 times at even intervals (`EVALUATIONS`); `--num_train_epochs` or `TRAIN_DOCUMENTS` changes the length. The header of `train_classifier.sh` has the details and an example.

Interrupted labelling runs resume from what they have written: resubmit the same command. A labelling job that completes no batch for 20 minutes (`--stall-timeout-minutes`, config `run.stall_timeout_minutes`) exits with code 1 and writes a traceback of where it hung to its `.err` log, instead of holding its GPUs until the time limit. On LUMI, the scripts in `hpc/lumi/` run the same code in a Singularity image; see the next section.

## Running on LUMI

The LUMI scripts load the `Local-LAIF` and `lumi-aif-singularity-bindings` modules and run Python inside `environments/lumi/vllm.sif`, which links to a LUMI AI Factory container. They bind `/scratch/project_462001516` and, for HPLT, `/appl/local/openeurollm`. Jobs are billed to `project_462001516`. Submit from `hpc/lumi/` so the relative log paths resolve; logs go to `logs/`.

HPLT 4.0 is read from `/appl/local/openeurollm/training/catalogue/hplt/4.0/<collection>/`, with one directory per language code. `sample_hplt_by_quality.sh` and `array_generate_distribution_hplt.sh` choose the collection or sample with the `data_quality` variable near their top: `clean`, `noisy` or `poor` for the stratified samples, `clean_random` and so on for the random ones.

```bash
cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi

# Random samples of 30,000 documents for each of the 128 languages of
# data/qwen_label_languages.txt with hplt-sampler: submit one run per language code
# (4 at a time), then collect them into data/hplt_<collection>_random_sample/
bash submit_hplt_sampler.sh clean 4
sbatch collect_hplt_samples.sh clean        # once the runs have finished; the same for noisy

# Stratified samples of 100,000 documents per language into data/hplt_<data_quality>_sample/
sbatch sample_hplt_by_quality.sh
sbatch sample_hplt_by_quality.sh --languages Latvian Latgalian "Norwegian Bokmål" "Norwegian Nynorsk" --overwrite

# Label the samples: 256 tasks, 128 languages x {topic, format}, 2 GPUs each
sbatch array_generate_distribution_hplt.sh [max_documents]

# One small test task first: Maltese (task 79), topic, 500 documents
sbatch --array=79 array_generate_distribution_hplt.sh 500

# Build a dataset per label type: per language 25,000 clean, 25,000 noisy, 10,000 poor documents
sbatch create_hplt_dataset.sh topic hplt-random-v1
sbatch create_hplt_dataset.sh format hplt-random-v1
```

- **Sampling** reads its languages from `data/oellm_languages.txt`; pass `--languages-file` for another list. A language whose sample file exists is skipped with a warning, so a rerun only adds missing languages; `--overwrite` replaces them. Language codes with no HPLT directory are reported as warnings, and the job still succeeds. Each language also gets a `<language>.manifest.json` with the files read and the documents taken per code, bucket and source.
- **hplt-sampler** ([TurkuNLP/hplt-sampler](https://github.com/TurkuNLP/hplt-sampler)) is cloned next to this repository, in `/scratch/project_462001516/users/tarkkaot/hplt-sampler`, with `hplt-sampler.conf` copied from `presets/lumi.conf` (`ACCOUNT` set). It samples each shard in proportion to its document count, reading every shard in full in parallel array jobs. Noisy runs need the fix on its branch `fix-collection-clash`, which keeps them apart from clean runs; `submit_hplt_sampler.sh` refuses to submit noisy runs without it. With several runs at once, set `BATCH` in `hplt-sampler.conf` to about 180 divided by their number, as they share the queue limits. Its per-file document counts (`counts/v4/<language>/<file>.json` for the clean collection) are also used to plan and check the classifier labelling.
- **Collecting** (`collect_hplt_samples.sh`) copies each language's run into `data/hplt_<collection>_random_sample/<language>.jsonl`. Every language of `data/qwen_label_languages.txt` has one code; with a list where a language has several, it gets 30,000 documents split over its codes in proportion to their document counts, then shuffled. A language with fewer than 30,000 documents in a collection gets all of them. It stops on an unfinished or unshuffled run, or a repeated document ID; `--allow-incomplete` collects the finished languages and skips the rest.
- **More random samples:** by default the sample is the first documents of each file, which can be dominated by a few websites. `--scan-factor N` draws them at random from the first N times as many documents as needed, and `--scan-factor 0` from all of them, at about N times the reading time. Pass a different `--output-dir` (or `--overwrite`) to keep it apart from the existing samples.
- **Labelling** uses `configs/lumi/distribution/generate_hplt.yaml` and always Qwen/Qwen3.8-27B, with tensor parallelism over the job's GPUs. It reads its languages from `languages_file` in the script, by default the 128 of `data/qwen_label_languages.txt`: task N is language N of the file (topic), task 128 + N its format labels. For the earlier 38-language clean, noisy and poor samples, set it to `data/oellm_languages.txt` and `--array` to `0-75`. Results go to `results/distribution/Qwen3.8-27B-hplt-annotations/<data_quality>/<label type>/<language>.jsonl`. Label the whole of a random sample, so that the labelled documents stay a random sample.
- **Speed:** a Maltese test labelled about 0.8–0.9 documents per second on 2 GPUs, so a task covers roughly 25,000 documents in its 8-hour limit. Resubmit a task to continue; it resumes where it stopped.
- **Datasets** (`create_hplt_dataset.sh`) run `create_datasets.py` on the clean, noisy and poor random results with `--max_per_file 25000 25000 10000`: the first documents of each result file, a random subset because the labelling follows the shuffled sample order. A file with fewer documents is used whole, with a warning in the log. A language's documents from the three collections are split together, 94/1/5 per language (`--stratify_by_language`; 1 % is still 600 validation documents per language), into `results/datasets/<label type>/<dataset_name>`. The Roihu launcher of the same name does the same. The `source` column (`clean_random`, `noisy_random` or `poor_random`) records the collection, and `finetune.py` reports test metrics per source as well as per language.
- `array_generate_WO_labels.sh` runs the earlier hard-label pipeline (`generate_multilabel.py`) on the parallel MT-Nemotron-CC documents.

### Labelling corpora with the classifiers

`classify_hplt.py` labels one HPLT 4.0 file (`<bucket>_<part>.jsonl.zst`) with the topic and format URL models: the same input as in training (`finetune.render_input`: percent-decoded URL, blank line, text, at most 8,192 tokens), bf16 autocast on fp32 weights. Two launchers share their settings in `classify_hplt_common.sh` (languages, collection, models, output root, `max_length`) and write the same output, so they can continue each other's files:

```bash
cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi

# Full standard-g nodes, 8 files at a time (one per GPU); the files are split into <nodes> sets of equal size
bash classify_hplt_node.sh list 20                 # the file sets of 20 nodes
sbatch --array=0-19 classify_hplt_node.sh 20       # default languages: deu_Latn spa_Latn fin_Latn

# One GPU per file on small-g (more than 210 tasks: submit_array_in_chunks.sh)
bash classify_hplt.sh list hun_Latn
sbatch --array=0-14 --time=48:00:00 classify_hplt.sh hun_Latn

# What is left: unfinished files and their task numbers, or resubmit them automatically
bash classify_hplt.sh pending [language ...]
bash submit_classify_leftovers.sh <array job ID>   # in tmux on a login node: waits for the array, then resubmits

# Label proportions of a finished language (CPU job, ~15-30 minutes for German)
sbatch label_proportions.sh deu_Latn
```

- **Resuming and checking:** results are appended to `<file>.jsonl.partial` and compressed under their final name only when every document is labelled, after checking one line per document. A rerun skips finished files and continues partial ones after checking their document IDs against the input. A lock file per input keeps two jobs from writing the same file; locks of jobs that are no longer in the queue are removed by the launchers.
- **Time limits:** the GPU runs at about 70,000 tokens per second per GCD. Estimate each file's time from its own tokens per document, which depend strongly on the quality bucket (see the status), and add about 15 minutes for resuming and up to an hour for compressing and checking. `submit_classify_leftovers.sh` estimates from per-language rates, which is too short for bucket 9 and 10 files.
- **Languages and task numbers:** the launchers take the languages as arguments (default: those in `classify_hplt_common.sh`). The task numbers and node sets depend on the languages, so resubmit with the same languages, and the same number of nodes for `classify_hplt_node.sh`.
- **Label proportions** (`label_proportions.sh`) refuse a language with unfinished files and check every output file against hplt-sampler's document count of its input. They write `results/statistics/classifier_label_proportions/hplt-4.0-clean/<language>.json` and `<language>.pairs.csv` (one row per topic × format pair).

## Output format

Each labelling run writes one JSONL line per document. The files are under `results/distribution/<model>-annotations/<label type>/<language>-parallel.jsonl`, or `<model>-hplt-annotations/<label type>/<language>.jsonl` for HPLT (on LUMI, `<model>-hplt-annotations/<data_quality>/<label type>/<language>.jsonl`).

| Field | Meaning |
|---|---|
| `id`, `text` | Document ID and the original, untruncated text |
| `url` | The URL shown to the model, or `null` if the document had none. It is read from the input's `url` field, or from `u` in HPLT records |
| `label`, `label_prob` | Most probable label (short name) and its probability |
| `label_probs`, `label_logprobs` | Probability and log-probability of every label, in label-file order |
| `label_order`, `example_order`, `prompt_order` | How labels and examples were ordered in the prompt |

Records are validated on writing; a run stops with an error instead of saving malformed results.

The classifier labels of HPLT (`classify_hplt.py`) are in `results/classifier_predictions/mmbert-1/hplt-4.0-<collection>/<language>/<bucket>_<part>.jsonl.zst`, one line per input document in input order. `metadata.json` in the collection directory records the models, the settings and the label order.

| Field | Meaning |
|---|---|
| `id`, `url`, `text` | HPLT's `id`, `u` and `text`, unchanged (the URL not percent-decoded, the text not truncated); HPLT's other fields are left out |
| `topic`, `format` | Most probable label of each classifier |
| `topic_probs`, `format_probs` | The full probability distribution of each classifier, rounded to 4 decimals, in the label order of `metadata.json` |

## Analyses

| Notebook | Question |
|---|---|
| `model_selection.ipynb` | Which open model should label the data? Consensus, calibration, consistency across languages, position bias, cost |
| `annotator_agreement.ipynb` | Agreement with the human annotator and the OpenAI models (κ, F1, bootstrap intervals); Clef against Qwen3.8-27B |
| `prompt_effect.ipynb` | Does our prompt change the labels compared with WebOrganizer's prompt? |
| `weborganizer_vs_propella.ipynb` | How do WebOrganizer's labels relate to the propella-1 annotations of the same documents? |
| `hplt_label_distributions.ipynb` | How do the label distributions differ between HPLT's clean, noisy and poor collections, and which languages stand out, and why? |
| `inspect_top_label_documents.ipynb` | What do the *Other* and *Bad example* documents look like? A random sample per language for reading by hand |
| `classifier_final.ipynb` | How good are the four final classifiers on the test split: overall, by source, language and label, the URL effect, calibration, against WebOrganizer, and the effect of duplicates of training documents |
| `classifier_half_epoch.ipynb` | The same analysis at half an epoch, on the validation split |
| `hp_search_results.ipynb` | Which hyperparameters work best? Ranks and plots the classifier searches |
| `model_comparison.ipynb` | Superseded by `model_selection.ipynb` |

## Documentation

- `docs/label_distribution_pipeline.md`: how the label distributions are produced, and how the pipeline differs from WebOrganizer's.
- `docs/labelling_model_recommendation.md`: the evidence for choosing Qwen3.8-27B.
- `docs/topic_annotation_guidelines.html`, `docs/format_annotation_guidelines.html`: the guidelines for human annotators, based word for word on the model prompt.
