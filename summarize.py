"""
Module 7 Week B — Integration Task: Summarization & Integrated Evaluation Report.

Implement the functions below. See the integration guide for full task descriptions.

The integrated evaluation report (`integrated-evaluation-report.md`) is the M7
deliverable. Write it by hand based on the metrics produced by main().
"""

import json
import os

import pandas as pd

from transformers import pipeline
from rouge_score import rouge_scorer

# -- Helpers (provided — do NOT modify) --------------------------------------

def get_summarizer_model_name() -> str:
    """Return env override (CI smoke) or the default summarization model."""
    return os.environ.get("SUMM_MODEL_FOR_CI", "sshleifer/distilbart-cnn-6-6")


def _articles_path() -> str:
    return os.environ.get("ARTICLES_PATH", "data/tech_news_articles.csv")


def _references_path() -> str:
    return os.environ.get("REFERENCES_PATH", "data/tech_news_summaries_reference.csv")


def _output_path() -> str:
    return os.environ.get("OUTPUT_PATH", "summary_predictions.csv")


# -- Task 1: Pipeline + single-document summarization ------------------------

def build_summarizer(model_name: str):
    """Construct a Hugging Face summarization pipeline."""
    # build a summarization pipeline using the given model name (same as the drill)
    return pipeline("summarization", model=model_name)


def summarize_one(summ, text: str, max_length: int = 120, min_length: int = 30) -> str:
    """
    Summarize one document with deterministic beam search.

    Use do_sample=False, num_beams=4. Return the summary STRING from
    [0]["summary_text"].
    """
    # invoke the pipeline with deterministic generation parameters (no sampling, beam search) and return the summary string
    output = summ(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        num_beams=4,
    )
    return output[0]["summary_text"]

# -- Task 2: ROUGE -----------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1.

    Use rouge_score.rouge_scorer.RougeScorer with use_stemmer=True.
    Argument order: scorer.score(reference, predicted) — REFERENCE FIRST.

    Returns {"rouge1": float, "rouge2": float, "rougeL": float}, all F1.
    """
    # build a stemming-enabled ROUGE scorer over the three metric variants
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    # score the (reference, predicted) pair and return F1 measures only (note argument order)
    scores = scorer.score(ref, pred)  # reference FIRST

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# -- Task 3: Evaluate over the corpus ----------------------------------------

def evaluate_summaries(summ, articles_df: pd.DataFrame, refs_df: pd.DataFrame) -> dict:
    """
    Summarize each article and score against its reference.

    Returns:
        {
          "rouge1": float, "rouge2": float, "rougeL": float,
          "n": int,
          "predictions": [
            {article_id, reference_summary, predicted_summary, rouge1, rouge2, rougeL},
            ...
          ],
        }

    Joins articles_df and refs_df on article_id.
    """
    #  merge the two DataFrames on article_id
    merged = pd.merge(articles_df, refs_df, on="article_id")

    predictions = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # iterate, summarize each article, compute ROUGE vs. reference
    for _, row in merged.iterrows():
        article_id = row["article_id"]
        text = row["text"]
        reference = row["reference_summary"]

        predicted = summarize_one(summ, text)
        rouge_scores = compute_rouge(predicted, reference)

        rouge1_scores.append(rouge_scores["rouge1"])
        rouge2_scores.append(rouge_scores["rouge2"])
        rougeL_scores.append(rouge_scores["rougeL"])

        predictions.append({
            "article_id": article_id,
            "reference_summary": reference,
            "predicted_summary": predicted,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
        })

    # aggregate (mean across summaries) and return the dict
    n = len(predictions)

    return {
        "rouge1": sum(rouge1_scores) / n if n > 0 else 0.0,
        "rouge2": sum(rouge2_scores) / n if n > 0 else 0.0,
        "rougeL": sum(rougeL_scores) / n if n > 0 else 0.0,
        "n": n,
        "predictions": predictions,
    }

# -- Task 4: Orchestrate -----------------------------------------------------

def main() -> None:
    """Load data, build pipeline, evaluate, write artifacts."""
    articles_df = pd.read_csv(_articles_path())
    refs_df = pd.read_csv(_references_path())

    summ = build_summarizer(get_summarizer_model_name())
    result = evaluate_summaries(summ, articles_df, refs_df)

    # Write predictions CSV
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(_output_path(), index=False)

    # Write metrics JSON
    metrics = {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "n": result["n"],
        "model": get_summarizer_model_name(),
    }
    metrics_path = _output_path().replace("predictions", "metrics").replace(".csv", ".json")
    if metrics_path == _output_path():  # safety: ensure rename happened
        metrics_path = "summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ROUGE-1 = {result['rouge1']:.4f}")
    print(f"ROUGE-2 = {result['rouge2']:.4f}")
    print(f"ROUGE-L = {result['rougeL']:.4f}")
    print(f"n = {result['n']}")


if __name__ == "__main__":
    main()
