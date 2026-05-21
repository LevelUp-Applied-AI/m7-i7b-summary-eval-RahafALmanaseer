
# Module 7 Week B — Integration Task: Summarization & Integrated Evaluation Report

This is the starter repo for the Module 7 Week B Integration Task. **The integrated evaluation report you produce here is the M7 deliverable.**

The full integration guide is at <a href="https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-7/496c1c2b" target="_blank">the integration guide page</a> — read it first.

## Quick start

```bash
pip install -r requirements.txt
make summarize    # runs full pipeline; first run downloads ~250 MB

```

The first call to `pipeline("summarization", ...)` downloads the model. Plan ~3 minutes for the first run; subsequent runs use cached weights. The full evaluation on 120 articles completes in ~6–8 minutes on CPU after the model is cached.

---

## Technical Model Documentation & Production Specifications

### 1. Production Model Identifier & Architecture

To implement a robust abstractive text summarization system, I configured this pipeline to leverage the sequence-to-sequence (Seq2Seq) bidirectional Transformer architecture through the fine-tuned model checkpoint `sshleifer/distilbart-cnn-6-6` hosted on the Hugging Face Hub. This specific checkpoint features a pruned DistilBART structure optimized for high throughput computational efficiency while maintaining high syntactic and semantic integrity. My pipeline architecture utilizes deterministic parameter control within the Hugging Face Transformers `pipeline("summarization")` extraction class.

### 2. Generation Parameter Controls

To eliminate stochastic variation and guarantee absolute replication across multi-stage continuous integration runs, I locked my summarization text extraction pipeline to a rigorous beam search inference configuration. The operational runtime specifies:

* `num_beams=4`: Tracks four concurrent hypothesis paths to construct high-probability sequences.
* `do_sample=False`: Disables random sampling entirely to force stable, deterministic token selection.
* `min_length`: Bound safely via dynamic defaults to encourage deep, cohesive synthesis.
* `max_length`: Capped structurally to match the reference summary window without introducing truncate errors.

### 3. Target Dataset & Corpus Context

I evaluated my system performance using a curated testing subset comprising exactly 120 technical, digital-culture, and business news articles parsed and extracted from the `glnmario/news-qa-summarization` raw domain pool. Each predicted sequence is scored directly against golden-standard professional targets stored inside `data/tech_news_summaries_reference.csv`. These baseline records represent actual human-authored editorial highlights written by native journalists, setting a rigorous lexical boundary for performance matching.

### 4. Deterministic Pipeline Replication Command

To execute, re-verify, and reproduce my absolute end-to-end evaluation execution flow—including document scoring and ROUGE performance computation—use the following terminal runtime configuration command:

```bash
ARTICLES_PATH=data/tech_news_articles.csv REFERENCES_PATH=data/tech_news_summaries_reference.csv OUTPUT_PATH=summary_predictions.csv python summarize.py

```

---

## What you will produce

Committed:

* `summarize.py` — your implementation
* Updated `README.md` — 1–2 paragraphs documenting model id, corpus version, re-run command (this section is the template; replace it)
* `summary_predictions.csv` — 120 rows with reference, predicted, and per-summary ROUGE
* `summary_metrics.json` — aggregate ROUGE-1/2/L F1
* `integrated-evaluation-report.md` — six-section integrated report (the M7 deliverable). Includes an optional Section 7 (Challenge Extensions) for learners completing challenge tiers — see the integration's learner guide.

**No model file** — pre-trained model loads from Hugging Face Hub at runtime.

## Data

* `data/tech_news_articles.csv` — 1,033 tech / entertainment / digital-culture news articles, curated from glnmario/news-qa-summarization. The full pool is here for inspection and stretch use; the integration evaluates on the 120-article subset that has reference summaries.
* `data/tech_news_summaries_reference.csv` — 120 reference summaries (one per evaluated article), shipped with the curated dataset (CNN editor-authored summaries from the source dataset).
* `data/tiny_articles_smoke.csv` + `data/tiny_refs_smoke.csv` — 3-row CI smoke fixtures (articles and references in separate files, matching the real-data schema).

## Make targets

```bash
make summarize    # full pipeline against the 120-article evaluation set
make smoke        # CI-only target — 3-row fixture
make clean        # remove generated outputs

```

## Submission

Open a Pull Request from your working branch into `main`. The autograder runs `make smoke` against the 3-row fixture and validates artifact schemas. PR description requirements are in the integration guide.

---

## License

This repository is provided for educational use only. See [LICENSE](https://www.google.com/search?q=LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.

```

```