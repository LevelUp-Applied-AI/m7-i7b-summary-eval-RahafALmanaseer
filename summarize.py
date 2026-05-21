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
    return pipeline("summarization", model=model_name, tokenizer=model_name)

def summarize_one(summ, text: str, max_length: int = 120, min_length: int = 30) -> str:
    """
    Summarize one document with deterministic beam search.

    Use do_sample=False, num_beams=4. Return the summary STRING from
    [0]["summary_text"].
    """
    # invoke the pipeline with deterministic generation parameters (no sampling, beam search) and return the summary string
    input_len = len(text.split())
    adj_max = min(max_length, max(10, input_len))
    adj_min = min(min_length, max(5, input_len // 2))
    
    if adj_min >= adj_max:
        adj_min = max(5, adj_max - 5)

    res = summ(
        text,
        max_length=adj_max,
        min_length=adj_min,
        do_sample=False,
        num_beams=4
    )
    return res[0]["summary_text"].strip() if res else ""

# -- Task 2: ROUGE -----------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1.

    Use rouge_score.rouge_scorer.RougeScorer with use_stemmer=True.
    Argument order: scorer.score(reference, predicted) — REFERENCE FIRST.

    Returns {"rouge1": float, "rouge2": float, "rougeL": float}, all F1.
    """
    # build a stemming-enabled ROUGE scorer over the three metric variants
    # score the (reference, predicted) pair and return F1 measures only (note argument order)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure)
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
    # merge the two DataFrames on article_id
    # iterate, summarize each article, compute ROUGE vs. reference
    # aggregate (mean across summaries) and return the dict

    merged_df = pd.merge(articles_df, refs_df, on="article_id")
    
    # Determine the reference summary column name dynamically (handles 'summary' or 'reference_summary')
    ref_col = "summary" if "summary" in merged_df.columns else "reference_summary"
    
    predictions = []
    r1_list, r2_list, rl_list = [], [], []
    
    for _, row in merged_df.iterrows():
        art_id = row["article_id"]
        text = row["text"]
        ref_summary = row[ref_col]  # Dynamic look-up here
        
        # Generate summary
        pred_summary = summarize_one(summ, text)
        
        # Compute ROUGE
        scores = compute_rouge(pred_summary, ref_summary)
        
        r1_list.append(scores["rouge1"])
        r2_list.append(scores["rouge2"])
        rl_list.append(scores["rougeL"])
        
        predictions.append({
            "article_id": art_id,
            "reference_summary": ref_summary,
            "predicted_summary": pred_summary,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"]
        })
        
    n = len(merged_df)
    
    return {
        "rouge1": float(sum(r1_list) / n) if n > 0 else 0.0,
        "rouge2": float(sum(r2_list) / n) if n > 0 else 0.0,
        "rougeL": float(sum(rl_list) / n) if n > 0 else 0.0,
        "n": n,
        "predictions": predictions
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
