# Multilingual WebOrganizer

Code for training multilingual versions of the [WebOrganizer](https://github.com/CodeCreator/WebOrganizer) topic and format classifiers. An open LLM labels web pages in many European languages with a **probability distribution** over 26 topic and 26 format labels. A multilingual encoder (mmBERT) is then trained on these soft labels, so that large web corpora can be organised by topic and format in every language.

The code runs on the CSC supercomputers Roihu and LUMI, through SLURM.

## Status (12 November 2026)

- **Labelling model chosen: Qwen3.8-27B.** It was compared with four other open models, a decision model built on it (Clef), two OpenAI models and a human annotator. The evidence is in `docs/labelling_model_recommendation.md`.
- **HPLT 4.0 labelled in 128 languages.** The languages and their codes are in `data/qwen_label_languages.txt`. Random samples of HPLT's clean, noisy and poor collections were labelled for topic and format on LUMI, and the results copied to Roihu: `results/distribution/Qwen3.8-27B-hplt-annotations/{clean,noisy,poor}_random/{topic,format}/<language>.jsonl`.
- **Training datasets built:** `results/datasets/{topic,format}/hplt-finetune` (input: URL and text) and `hplt-finetune-nourl` (text only; otherwise identical). Each has 7,288,056 documents: up to 25,000 clean, 25,000 noisy and 10,000 poor documents per language, split 94 / 1 / 5 % per language into train (6,850,772), validation (72,882) and test (364,402). 17 low-resource languages have fewer clean or noisy documents: Magahi, Banjar (Arabic and Latin), Kabuverdianu, Pangasinan, Tok Pisin, Chhattisgarhi, Latgalian, Tibetan, Waray, Awadhi, Ligurian, Balinese, Minangkabau, Sicilian, Silesian and Maithili. Documents and mmBERT tokens per language: `results/statistics/hplt-random.csv` (`count_language_tokens.sh`).
- **Duplicate check** (`find_duplicates.sh`, results in `results/statistics/duplicates/`): exact duplicates are negligible, and 0.3 % of training documents are near-duplicate copies, concentrated in Traditional Chinese SEO spam, Tibetan, Tok Pisin and Maithili. 0.47 % of validation and test documents are near-duplicates of training documents; leaving them out of the half-epoch validation results changes overall accuracy by at most 0.03 points. The URL models show no sign of memorising URLs that recur across splits.
- **Now: training four classifiers** (mmBERT-base; topic and format, each with and without the URL) on Roihu, started on 11 November: `results/models/<label type>/<dataset>/mmbert-{topic,format}[-nourl]-1`. Each is one epoch (428,174 steps of 16 documents) with the best settings of the hyperparameter searches (learning rate 3.05e-5, weight decay 0.088, warmup 7.6 %, mean pooling), evaluated and checkpointed 20 times. On 12 November at 13:00 they were 65–73 % through, at about 6–9 steps per second, so each should finish within its first 36-hour job; the chained second job then only repeats the final evaluation and can be cancelled.
- **Half-epoch results** (step 214,090, validation split; `notebooks/classifier_half_epoch.ipynb`): agreement with Qwen3.8-27B's top label is 86.0 % for topic and 86.5 % for format with the URL, against 83.4 % and 83.2 % without it. The URL helps in every measure, and in no language is the text-only model clearly better. On WebOrganizer's confident-subset measure (the labeller's top label above 75 % probability, the 24 labels both share), format already exceeds WebOrganizer's reported English classifiers (label average 94.4 % vs 91.8 %) and topic is close (92.1 % vs 93.5 %).
- **Next:** evaluate the final models on the test split without the duplicated documents (a copy of `classifier_half_epoch.ipynb` with `SPLIT = "test"`, `STEP = None`, `EXCLUDE_LEAKED = True`); check against the human annotations whether the URL also helps there, since Qwen3.8-27B labelled with the URL in its prompt; for the next dataset version, split by duplicate cluster so that duplicates cannot cross splits.
- **First classifiers (October 2026)** were trained on `hplt-v1`: 5,000 documents for each of 36 European languages, 144,145 training documents. On the test split, topic reached accuracy 0.860 (Jensen–Shannon divergence 0.091 bits) and format 0.847 (0.063 bits).

## How it works

1. **Sample documents.** `scripts/data_processing/sample_hplt_by_quality.py` draws a sample per language from one of HPLT 4.0's collections (clean, noisy or poor), stratified by HPLT's quality bucket. The languages come from a file of `Language: code [code ...]` lines. The sample is split evenly over a language's codes, then its quality buckets, then the sources within a bucket. By default the documents are taken from the start of each file; with `--scan-factor N` they are drawn at random from the first N times as many as needed (`0`: from the whole source). The files hold their documents grouped by crawl, so the start of a file is not representative. `--method bernoulli` therefore instead keeps every document of every file with the same probability, with no quotas, so the sample follows the composition of the full data. A document whose ID is already in the sample is skipped, and the output is shuffled. It runs on LUMI via `hpc/lumi/sample_hplt_by_quality.sh`. The Roihu copy is in `data/hplt4/<language>.jsonl`: 5,000 documents for each of 36 languages.
2. **Label them with label distributions.** `scripts/data_processing/generate_label_distributions.py` shows the model the label list under single-letter IDs, with worked examples, and reads the probabilities of the answer letters in one forward pass (vLLM). Each document gets a probability for every label, normalised the same way as in WebOrganizer.
3. **Build training data.** `scripts/model_training/create_datasets.py` turns the label files into train, validation and test splits. It splits by document ID, so the translations of a document never end up in different splits. With `--stratify_by_language`, every language is split in exactly the requested ratios instead of approximately, which matters for small languages.
4. **Train the classifier.** `scripts/model_training/finetune.py` fine-tunes mmBERT (`jhu-clsp/mmBERT-base`) with cross-entropy against the label distributions. As in WebOrganizer's default classifiers, the input is the URL, a blank line, and the text; URLs are percent-decoded first. A dataset made with `create_datasets.py --no_url` has no URLs and gives a text-only classifier. `--template` overrides the input, and the template is saved in the model's `config.json`. `hp_search.py` and `summarize_hp_search.py` run a random hyperparameter search, one trial per SLURM array task.
5. **Evaluate.** `scripts/data_analysis/` and `notebooks/` compare models with each other, across languages, with WebOrganizer's labels and with human labels.

### Labels

The label definitions are in `scripts/data_processing/topics.yaml` and `formats.yaml`, and the worked examples in `topic_examples.yaml` and `format_examples.yaml`. Each set has WebOrganizer's 24 labels plus two fallbacks:

- **Other:** the page is clearly outside all other labels.
- **Bad example:** the content is unintelligible or too short to label.

### Prompt order

To keep the model from favouring answer positions, the label and example order changes per document. With `--prompt-order rotated`, documents share one of 26 rotations, so vLLM can reuse the cached start of the prompt; this cuts labelling cost substantially. Each language uses its own seed, so a position preference can't make translations look more consistent than they are.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/data_processing/` | Sampling, labelling (`generate_label_distributions.py`, plus `generate_label_clef.py` for Clef and `generate_openai_labels.py` for OpenAI models), prompts, labels, the human-evaluation sample, and shared code (`labeling_common.py`, `distribution_records.py`) |
| `scripts/model_training/` | Dataset creation, classifier fine-tuning, hyperparameter search |
| `scripts/data_analysis/` | Evaluation: consistency across languages, agreement with WebOrganizer, shared metrics (`distribution_metrics.py`) |
| `hpc/roihu/`, `hpc/lumi/` | SLURM launch scripts; each documents its usage in its header |
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
```

`train_classifier.sh` reports to Weights & Biases by default (project `multilingual-weborganizer-finetune`); run `wandb login` once on a login node first. Submit with `WANDB=offline` to sync later with `wandb sync`, or `WANDB=` to turn reporting off. `gpumedium` jobs run for at most 36 hours, and a run resumes from its last checkpoint when resubmitted, so long runs are chained jobs. By default a run is one epoch, evaluated and checkpointed 20 times at even intervals (`EVALUATIONS`); `--num_train_epochs` or `TRAIN_DOCUMENTS` changes the length. The header of `train_classifier.sh` has the details and an example.

Interrupted labelling runs resume from what they have written: resubmit the same command. A labelling job that completes no batch for 20 minutes (`--stall-timeout-minutes`, config `run.stall_timeout_minutes`) exits with code 1 and writes a traceback of where it hung to its `.err` log, instead of holding its GPUs until the time limit. On LUMI, the scripts in `hpc/lumi/` run the same code in a Singularity image; see the next section.

## Running on LUMI

The LUMI scripts load the `Local-LAIF` and `lumi-aif-singularity-bindings` modules and run Python inside `environments/lumi/vllm.sif`, which links to a LUMI AI Factory container. They bind `/scratch/project_462001516` and, for HPLT, `/appl/local/openeurollm`. Jobs are billed to `project_462001516`. Submit from `hpc/lumi/` so the relative log paths resolve; logs go to `logs/`.

HPLT 4.0 is read from `/appl/local/openeurollm/training/catalogue/hplt/4.0/<collection>/`, with one directory per language code. Both HPLT scripts choose the collection with the `data_quality` variable near their top: `clean`, `noisy` or `poor`.

```bash
cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi

# Random samples of 30,000 documents for each of the 128 languages of
# data/qwen_label_languages.txt with hplt-sampler: submit one run per language code
# (4 at a time), then collect them into data/hplt_<collection>_random_sample/
bash submit_hplt_sampler.sh clean 4
sbatch collect_hplt_samples.sh clean        # once the runs have finished; the same for noisy

# Or, without hplt-sampler, Bernoulli samples: one task per language and collection
sbatch sample_hplt_bernoulli.sh

# Then reuse earlier labels for them (once all samples are ready)
sbatch reuse_hplt_labels.sh

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
- **hplt-sampler** ([TurkuNLP/hplt-sampler](https://github.com/TurkuNLP/hplt-sampler)) is cloned next to this repository, in `/scratch/project_462001516/users/tarkkaot/hplt-sampler`, with `hplt-sampler.conf` copied from `presets/lumi.conf` (`ACCOUNT` set). It samples each shard in proportion to its document count, reading every shard in full in parallel array jobs. Noisy runs need the fix on its branch `fix-collection-clash`, which keeps them apart from clean runs; `submit_hplt_sampler.sh` refuses to submit noisy runs without it. With several runs at once, set `BATCH` in `hplt-sampler.conf` to about 180 divided by their number, as they share the queue limits.
- **Collecting** (`collect_hplt_samples.sh`) copies each language's run into `data/hplt_<collection>_random_sample/<language>.jsonl`. Every language of `data/qwen_label_languages.txt` has one code; with a list where a language has several, it gets 30,000 documents split over its codes in proportion to their document counts, then shuffled. A language with fewer than 30,000 documents in a collection gets all of them. It stops on an unfinished or unshuffled run, or a repeated document ID; `--allow-incomplete` collects the finished languages and skips the rest.
- **Bernoulli sampling** (`sample_hplt_bernoulli.sh`) reads every file in full, twice: once to count the documents and set the probability to draw slightly more than 30,000, once to sample. The drawn documents are shuffled and the first 30,000 kept, so each sample has exactly 30,000 documents and is a simple random sample of the language. Each task reads its files with one worker per CPU; the 24-hour limit is a guess, so check the first tasks before running all 76.
- **Reusing earlier labels:** after the random samples are made and before labelling them, `sbatch reuse_hplt_labels.sh` copies the existing results whose document ID, text and URL match a sample document into `results/distribution/Qwen3.8-27B-hplt-annotations/{clean,noisy}_random/`, and moves those documents to the top of the sample file. Labelling with `data_quality=clean_random` (or `noisy_random`) then skips them. Label the whole sample: the reused documents are not a random subset, so a run that stops early is tilted towards them.
- **More random samples:** by default the sample is the first documents of each file, which can be dominated by a few websites. `--scan-factor N` draws them at random from the first N times as many documents as needed, and `--scan-factor 0` from all of them, at about N times the reading time. Pass a different `--output-dir` (or `--overwrite`) to keep it apart from the existing samples.
- **Labelling** uses `configs/lumi/distribution/generate_hplt.yaml` and always Qwen/Qwen3.8-27B, with tensor parallelism over the job's GPUs. It reads its languages from `languages_file` in the script, by default the 128 of `data/qwen_label_languages.txt`: task N is language N of the file (topic), task 128 + N its format labels. For the earlier 38-language clean, noisy and poor samples, set it to `data/oellm_languages.txt` and `--array` to `0-75`. Results go to `results/distribution/Qwen3.8-27B-hplt-annotations/<data_quality>/<label type>/<language>.jsonl`.
- **Speed:** a Maltese test labelled about 0.8–0.9 documents per second on 2 GPUs, so a task covers roughly 25,000 documents in its 8-hour limit. Resubmit a task to continue; it resumes where it stopped.
- **Datasets** (`create_hplt_dataset.sh`) run `create_datasets.py` on the clean, noisy and poor random results with `--max_per_file 25000 25000 10000`: the first documents of each result file, a random subset because the labelling follows the shuffled sample order. A file with fewer documents is used whole, with a warning in the log. A language's documents from the three collections are split together, 94/1/5 per language (`--stratify_by_language`; 1 % is still 600 validation documents per language), into `results/datasets/<label type>/<dataset_name>`. The Roihu launcher of the same name does the same. The `source` column (`clean_random`, `noisy_random` or `poor_random`) records the collection, and `finetune.py` reports test metrics per source as well as per language.
- `array_generate_WO_labels.sh` runs the earlier hard-label pipeline (`generate_multilabel.py`) on the parallel MT-Nemotron-CC documents.

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

## Analyses

| Notebook | Question |
|---|---|
| `model_selection.ipynb` | Which open model should label the data? Consensus, calibration, consistency across languages, position bias, cost |
| `annotator_agreement.ipynb` | Agreement with the human annotator and the OpenAI models (κ, F1, bootstrap intervals); Clef against Qwen3.8-27B |
| `prompt_effect.ipynb` | Does our prompt change the labels compared with WebOrganizer's prompt? |
| `weborganizer_vs_propella.ipynb` | How do WebOrganizer's labels relate to the propella-1 annotations of the same documents? |
| `classifier_half_epoch.ipynb` | How do the four classifiers (topic and format, with and without the URL) compare at half an epoch: overall, by source, language and label, calibration, against WebOrganizer's reported accuracy, and with duplicates of training documents left out. Reusable for the final models (see its introduction) |
| `hp_search_results.ipynb` | Which hyperparameters work best? Ranks and plots the classifier searches |
| `model_comparison.ipynb` | Superseded by `model_selection.ipynb` |

## Documentation

- `docs/label_distribution_pipeline.md`: how the label distributions are produced, and how the pipeline differs from WebOrganizer's.
- `docs/labelling_model_recommendation.md`: the evidence for choosing Qwen3.8-27B.
- `docs/topic_annotation_guidelines.html`, `docs/format_annotation_guidelines.html`: the guidelines for human annotators, based word for word on the model prompt.
