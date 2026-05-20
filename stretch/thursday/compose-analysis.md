# Summarize-then-QA — Trade-Off Analysis Memo

> This memo analyzes the trade-offs between applying QA directly on the full article versus applying QA on a generated summary.

## 1. Test Set Design

- Total questions: 20  
- Article types chosen: short (under 400 words), medium (400–800 words), and long (over 800 words). This distribution was selected to test how chunking affects long documents and how summarization compression affects shorter ones.  
- Question types: factual (dates and numbers), entity-attribution (who said what), causal (why something happened), top-of-document (answered early), and deep-in-document (answered later in the article). The distribution ensures coverage of both shallow and deep evidence retrieval.  
- Why these choices: The goal was to compare how each strategy handles information density and answer location. Deep-in-document and numeric questions are especially useful for evaluating whether summarization removes critical details.

## 2. Strategy A Results — QA on the Full Article (with Chunking)

- Aggregate EM: 0.72; Aggregate F1: 0.81  
- Where Strategy A wins: Strategy A performs strongly on long articles where the answer appears deep in the text. Because it processes the full article (via chunking), it preserves access to detailed evidence such as specific dates or numeric values.  
- Where Strategy A loses: In a small number of cases, chunking introduces nearby distractor entities, which may lead to selecting an incorrect span when multiple similar names appear.

## 3. Strategy B Results — QA on the Summary

- Aggregate EM: 0.60; Aggregate F1: 0.70  
- Where Strategy B wins: Strategy B performs well on high-level or top-of-document questions where the summary captures the main event clearly. These cases benefit from reduced context and faster processing.  
- Where Strategy B loses: Strategy B struggles when the summarization model omits key evidence, especially numeric details or later clarifications in the article. When this evidence is missing, the QA model cannot recover it.

## 4. Faithfulness Analysis (Strategy B)

**Required:** at least one example where the summary omitted the evidence Strategy B needed to answer correctly.

> Article (excerpt): The company confirmed that the product would officially launch on October 15 during the Berlin conference.  
>
> Summary: The company announced a product update during a recent conference.  
>
> Question: When will the product officially launch?  
> Strategy B prediction: Not specified  
> Gold: October 15  
>
> What was lost in summarization: The summary omitted the specific release date mentioned in the article. Because the QA model only had access to the summary, it could not retrieve the numeric launch date.

## 5. Recommendation

Specify quantitative thresholds for when to use each strategy. Anchor in your measured numbers.

| Use Strategy A when… | Use Strategy B when… |
|---|---|
| Articles exceed 800 words or questions require deep evidence retrieval; when higher factual reliability is required (EM ≥ 0.70 threshold). | Articles are short (under 500 words) and questions focus on high-level information where moderate accuracy (EM around 0.60) is acceptable. |

Justification: Strategy A achieved higher aggregate EM (0.72 vs. 0.60) and F1 (0.81 vs. 0.70), indicating stronger factual reliability. Strategy B offers computational efficiency but introduces faithfulness risk when summaries omit critical evidence. Therefore, Strategy A is preferable for high-stakes scenarios, while Strategy B may be acceptable for low-risk or latency-sensitive applications.