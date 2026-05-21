"""
Module 7 Week B — Thursday Stretch (Honors): Summarize-then-QA.

Composes the Integration 7B summarizer with the Lab 7B QA pipeline. You will
need the QA pipeline + EM/F1 functions from your Lab 7B `lab.py` — copy them
into `qa_utils.py` here, or import them via a path stub (see TODO below).

Implement the four TODO functions; see the stretch page for full task description.
"""

import json
import os
import sys

import pandas as pd

# Import the integration's summarizer functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import summarize  # noqa: E402

# QA pipeline + EM/F1 functions are NOT in this Integration 7B repo. Copy them
# from your Lab 7B `lab.py` into `stretch/thursday/qa_utils.py` so this script
# can import them. The autograder verifies these functions are callable; do not
# proceed without them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import qa_utils  # noqa: E402  (provides build_qa_pipeline, predict_one, exact_match, token_f1, normalize_answer, get_qa_model_name)
except ImportError as e:
    raise ImportError(
        "qa_utils.py not found. Copy build_qa_pipeline, predict_one, exact_match, "
        "token_f1, normalize_answer, and get_qa_model_name from your Lab 7B lab.py "
        "into stretch/thursday/qa_utils.py."
    ) from e


def qa_full_article(qa, question: str, article: str, max_chunk: int = 384) -> str:
    """
    Run QA over the full article, chunking with overlap when it exceeds max_chunk tokens.

    Returns the answer span from the highest-scoring chunk.
    """
    # token-count the article (use a tokenizer or a rough word count); if it fits, call qa once and return predict_one's answer
    # if it exceeds max_chunk, split into overlapping windows (e.g., 384-token windows with 64-token overlap)
    # call qa on each window; track the pipeline's score per window
    # return the answer string from the highest-scoring window

    tokenizer = qa.tokenizer
    
    tokens = tokenizer.encode(article, add_special_tokens=False)
    
    if len(tokens) <= max_chunk:
        return qa_utils.predict_one(qa, question, article)
    
    stride = 64
    best_answer = ""
    best_score = -float("inf")
    
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_chunk, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        res = qa(question=question, context=chunk_text)
        
        if res and "score" in res:
            current_score = res["score"]
            if current_score > best_score:
                best_score = current_score
                best_answer = res.get("answer", "")
        
        if end_idx == len(tokens):
            break
            
        start_idx += (max_chunk - stride)
        
    return best_answer

def qa_via_summary(qa, summ, question: str, article: str, max_summary_length: int = 120) -> str:
    """
    Summarize the article first, then run QA on the summary.

    Returns the answer string. Uses Integration 7B's summarize_one (do_sample=False, num_beams=4).
    """
    # summarize the article using summarize.summarize_one with the given max_summary_length
    # run QA on the summary using qa_utils.predict_one
    # return the answer string

    summary_text = summarize.summarize_one(
        summ, 
        article, 
        max_length=max_summary_length
    )
    
    answer = qa_utils.predict_one(qa, question, summary_text)
    
    return answer

def evaluate_strategies(qa, summ, test_set: pd.DataFrame, articles_df: pd.DataFrame) -> dict:
    """
    Run both strategies on every row of the test set; compute per-strategy EM/F1.

    Returns:
        {
          "strategy_a": {"em": float, "f1": float, "n": int},
          "strategy_b": {"em": float, "f1": float, "n": int},
          "predictions": [
            {qid, question, strategy_a_pred, strategy_b_pred, gold_answer,
             strategy_a_em, strategy_a_f1, strategy_b_em, strategy_b_f1},
            ...
          ],
        }
    """
    # for each test_set row, look up the article in articles_df by article_id
    # call qa_full_article (Strategy A) and qa_via_summary (Strategy B); record predictions
    # compute EM + F1 for each strategy via qa_utils.exact_match / qa_utils.token_f1
    # aggregate per-strategy means; return the combined dict
    predictions = []
    
    articles_dict = dict(zip(articles_df["article_id"], articles_df["text"]))
    
    strategy_a_em_list = []
    strategy_a_f1_list = []
    strategy_b_em_list = []
    strategy_b_f1_list = []
    
    for _, row in test_set.iterrows():
        qid = row["qid"]
        article_id = row["article_id"]
        question = row["question"]
        gold_answer = row["gold_answer"]
        
        article_text = articles_dict.get(article_id, "")
        
        pred_a = qa_full_article(qa, question, article_text)
        
        pred_b = qa_via_summary(qa, summ, question, article_text)
        
        em_a = qa_utils.exact_match(pred_a, gold_answer)
        f1_a = qa_utils.token_f1(pred_a, gold_answer)
        
        em_b = qa_utils.exact_match(pred_b, gold_answer)
        f1_b = qa_utils.token_f1(pred_b, gold_answer)
        
        strategy_a_em_list.append(em_a)
        strategy_a_f1_list.append(f1_a)
        strategy_b_em_list.append(em_b)
        strategy_b_f1_list.append(f1_b)
        
        predictions.append({
            "qid": qid,
            "question": question,
            "strategy_a_pred": pred_a,
            "strategy_b_pred": pred_b,
            "gold_answer": gold_answer,
            "strategy_a_em": float(em_a),
            "strategy_a_f1": float(f1_a),
            "strategy_b_em": float(em_b),
            "strategy_b_f1": float(f1_b)
        })
        
    n_total = len(test_set)
    
    results = {
        "strategy_a": {
            "em": float(sum(strategy_a_em_list) / n_total) if n_total > 0 else 0.0,
            "f1": float(sum(strategy_a_f1_list) / n_total) if n_total > 0 else 0.0,
            "n": n_total
        },
        "strategy_b": {
            "em": float(sum(strategy_b_em_list) / n_total) if n_total > 0 else 0.0,
            "f1": float(sum(strategy_b_f1_list) / n_total) if n_total > 0 else 0.0,
            "n": n_total
        },
        "predictions": predictions
    }
    
    return results

def main() -> None:
    """Load test set + articles, build pipelines, run both strategies, write artifacts."""
    test_set = pd.read_csv("stretch/thursday/qa_test_set.csv")
    articles_df = pd.read_csv("data/tech_news_articles.csv")

    qa = qa_utils.build_qa_pipeline(qa_utils.get_qa_model_name())
    summ = summarize.build_summarizer(summarize.get_summarizer_model_name())

    result = evaluate_strategies(qa, summ, test_set, articles_df)

    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("stretch/thursday/compose_predictions.csv", index=False)

    metrics = {
        "strategy_a": result["strategy_a"],
        "strategy_b": result["strategy_b"],
        "qa_model": qa_utils.get_qa_model_name(),
        "summarizer_model": summarize.get_summarizer_model_name(),
    }
    with open("stretch/thursday/compose_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Strategy A (full-article QA) — EM={result['strategy_a']['em']:.4f}, F1={result['strategy_a']['f1']:.4f}")
    print(f"Strategy B (summarize-then-QA) — EM={result['strategy_b']['em']:.4f}, F1={result['strategy_b']['f1']:.4f}")


if __name__ == "__main__":
    main()
