import re
import string
import torch
from transformers import pipeline

def get_qa_model_name() -> str:
    """Return the QA model name."""
    return "distilbert-base-cased-distilled-squad"

def build_qa_pipeline(model_name: str):
    """Build the Hugging Face QA pipeline."""
    return pipeline("question-answering", model=model_name, tokenizer=model_name)

def predict_one(qa_pipeline, question: str, context: str) -> str:
    """Get the answer string from the QA pipeline."""
    res = qa_pipeline(question=question, context=context)
    return res.get("answer", "") if res else ""

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra spaces."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction: str, gold: str) -> int:
    """Compute exact match score."""
    return int(normalize_answer(prediction) == normalize_answer(gold))

def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
        
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)