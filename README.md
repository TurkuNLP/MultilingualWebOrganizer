# Multilingual WebOrganizer

Code for training multilingual versions of the [WebOrganizer](https://github.com/CodeCreator/WebOrganizer) topic and format classifiers. An open LLM labels web pages in many European languages with a **probability distribution** over 26 topic and 26 format labels. A multilingual encoder (mmBERT) is then trained on these soft labels, so that large web corpora can be organised by topic and format in every language.

The code runs on the CSC supercomputers Roihu and LUMI, through SLURM.

## Status (October 2026)

- **Labelling model chosen: Qwen3.8-27B.** It was compared with four other open models, a decision model built on it (Clef), two OpenAI models and a human annotator. The evidence is in `docs/labelling_model_recommendation.md`.
- **HPLT 4.0 samples are ready on LUMI** for the OpenEuroLLM languages: 100,000 documents per language from HPLT's clean and noisy collections, and 10,000 from the poor collection. Latvian and Norwegian were since split into Latvian and Latgalian, and Norwegian Bokmål and Nynorsk; these four still need sampling under their new names.
- **Next: labelling these samples** with Qwen3.8-27B on LUMI.
- **Proposed wider language set: 125 languages** in `data/qwen_label_languages.txt`. These are the OpenEuroLLM languages plus languages on Qwen3's official list or in mmBERT's pre- and mid-training data. They have not been sampled yet.
- **Classifier training** (`scripts/model_training/`) is implemented but has not been run yet.

## How it works

1. **Sample documents.** `scripts/data_processing/sample_hplt_by_quality.py` draws a sample per language from one of HPLT 4.0's collections (clean, noisy or poor), stratified by HPLT's quality bucket. The languages come from a file of `Language: code [code ...]` lines. The sample is split evenly over a language's codes, then its quality buckets, then the sources within a bucket, taken from the start of each file. A document whose ID is already in the sample is skipped, and the output is shuffled. It runs on LUMI via `hpc/lumi/sample_hplt_by_quality.sh`. The Roihu copy is in `data/hplt4/<language>.jsonl`: 5,000 documents for each of 36 languages.
2. **Label them with label distributions.** `scripts/data_processing/generate_label_distributions.py` shows the model the label list under single-letter IDs, with worked examples, and reads the probabilities of the answer letters in one forward pass (vLLM). Each document gets a probability for every label, normalised the same way as in WebOrganizer.
3. **Build training data.** `scripts/model_training/create_datasets.py` turns the label files into train, validation and test splits. It splits by document ID, so the translations of a document never end up in different splits.
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
| `data/`, `results/`, `logs/` | Inputs, outputs and job logs (not version-controlled). Language lists: `data/oellm_languages.txt` (the 38 OpenEuroLLM languages), `data/hplt_languages.txt` (all 187 HPLT 4.0 languages) and `data/qwen_label_languages.txt` (the proposed 125) |

## Running on Roihu

Scripts load the `python-vllm/0.19.1` module, which provides Python 3.12, vLLM 0.19.1, torch and transformers 5.8. Submit from `hpc/roihu/` so the relative log paths resolve; logs go to `logs/`.

```bash
cd /scratch/project_2020507/users/tarkkaot/MultilingualWebOrganizer/hpc/roihu

# Label the HPLT samples: 72 tasks, 36 languages x {topic, format}
sbatch array_generate_distribution_hplt.sh [max_documents]

# Label the parallel MT-Nemotron-CC evaluation documents (6 languages x 2 label types)
sbatch array_generate_distribution.sh Qwen/Qwen3.8-27B 2000

# Build a dataset from a labelling run, then train (or search hyperparameters)
sbatch create_datasets.sh Qwen3.8-27B-hplt-annotations topic hplt-v1
sbatch train_classifier.sh topic hplt-v1 mmbert
sbatch --array=0-15%4 hp_search_classifier.sh topic hplt-v1 search1
```

Interrupted labelling runs resume from what they have written: resubmit the same command. A labelling job that completes no batch for 20 minutes (`--stall-timeout-minutes`, config `run.stall_timeout_minutes`) exits with code 1 and writes a traceback of where it hung to its `.err` log, instead of holding its GPUs until the time limit. On LUMI, the scripts in `hpc/lumi/` run the same code in a Singularity image; see the next section.

## Running on LUMI

The LUMI scripts load the `Local-LAIF` and `lumi-aif-singularity-bindings` modules and run Python inside `environments/lumi/vllm.sif`, which links to a LUMI AI Factory container. They bind `/scratch/project_462001516` and, for HPLT, `/appl/local/openeurollm`. Jobs are billed to `project_462001516`. Submit from `hpc/lumi/` so the relative log paths resolve; logs go to `logs/`.

HPLT 4.0 is read from `/appl/local/openeurollm/training/catalogue/hplt/4.0/<collection>/`, with one directory per language code. Both HPLT scripts choose the collection with the `data_quality` variable near their top: `clean`, `noisy` or `poor`.

```bash
cd /scratch/project_462001516/users/tarkkaot/MultilingualWebOrganizer/hpc/lumi

# Sample 100,000 documents per language into data/hplt_<data_quality>_sample/
sbatch sample_hplt_by_quality.sh
sbatch sample_hplt_by_quality.sh --languages Latvian Latgalian "Norwegian Bokmål" "Norwegian Nynorsk" --overwrite

# Label the samples: 76 tasks, 38 languages x {topic, format}, 2 GPUs each
sbatch array_generate_distribution_hplt.sh [max_documents]

# One small test task first: Maltese (task 15), topic, 500 documents
sbatch --array=15 array_generate_distribution_hplt.sh 500
```

- **Sampling** reads its languages from `data/oellm_languages.txt`; pass `--languages-file` for another list. A language whose sample file exists is skipped with a warning, so a rerun only adds missing languages; `--overwrite` replaces them. Language codes with no HPLT directory are reported as warnings, and the job still succeeds. Each language also gets a `<language>.manifest.json` with the files read and the documents taken per code, bucket and source.
- **Labelling** uses `configs/lumi/distribution/generate_hplt.yaml` and always Qwen/Qwen3.8-27B, with tensor parallelism over the job's GPUs. Its 38 languages are written into the script under their sample-file names (e.g. `norwegian_bokmål`); with a different language list, update the list and `--array` (2 × the number of languages). Results go to `results/distribution/Qwen3.8-27B-hplt-annotations/<data_quality>/<label type>/<language>.jsonl`.
- **Speed:** a Maltese test labelled about 0.8–0.9 documents per second on 2 GPUs, so a task covers roughly 25,000 documents in its 8-hour limit. Resubmit a task to continue; it resumes where it stopped.
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
| `model_comparison.ipynb` | Superseded by `model_selection.ipynb` |

## Documentation

- `docs/label_distribution_pipeline.md`: how the label distributions are produced, and how the pipeline differs from WebOrganizer's.
- `docs/labelling_model_recommendation.md`: the evidence for choosing Qwen3.8-27B.
- `docs/topic_annotation_guidelines.html`, `docs/format_annotation_guidelines.html`: the guidelines for human annotators, based word for word on the model prompt.
