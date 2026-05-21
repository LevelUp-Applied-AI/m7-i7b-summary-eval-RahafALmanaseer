# Summarize-then-QA — Trade-Off Analysis Memo

> Replace each placeholder section. Memo target: ~1.5 pages. The TA rubric rewards quantitative thresholds and concrete examples.

## 1. Test Set Design

- Total questions: 20
- Article types chosen: I chose a mix of short, medium, and long-form technical news articles ranging from 300 to 1,200 words. Including longer articles was essential to test how well my sliding-window chunking logic could handle text that exceeds the standard model context limits.
- Question types: I distributed my questions to cover 40% factual metrics, 30% entity-attribution, 20% causal reasoning, and a 10% mix of top-of-document vs. deep-in-document information lookups.
- Why these choices: I made these design choices to thoroughly stress-test both components of my pipeline. Technical engineering news often embeds critical numbers, timelines, or metrics deep within long paragraphs, making it the perfect benchmark to evaluate where my summarization model succeeds in preserving facts and where it drops fine-grained details needed by my QA model.

## 2. Strategy A Results — QA on the Full Article (with Chunking)

- Aggregate EM: 0.8000; Aggregate F1: 0.8200
- Where Strategy A wins: Strategy A performed exceptionally well on deep-in-document lookups like `qid_003` and `qid_007`. Because my implementation processes the raw text directly through a sliding window, the QA pipeline maintained direct access to exact verbatim numbers and original metrics without any loss of detail.
- Where Strategy A loses: Strategy A struggled on `qid_012`, where the answer was surrounded by similar technical metrics. The chunking boundary introduced slight context noise, and the model picked a distractor entity from an overlapping window instead of the true gold target.

## 3. Strategy B Results — QA on the Summary

- Aggregate EM: 0.1500; Aggregate F1: 0.1950
- Where Strategy B wins: Strategy B managed to successfully extract correct answers for high-level or top-of-document questions like `qid_001` and `qid_002`. These cases involved general concepts that my summarizer naturally kept during its main compression pass.
- Where Strategy B loses: Strategy B failed completely on specific, granular queries like `qid_015` and `qid_019`. This major drop happened because my abstractive summarizer prioritized the broader theme of the text and discarded the specific dates and numerical figures that my downstream QA model needed to successfully match the gold targets.

## 4. Faithfulness Analysis (Strategy B)

**Required:** at least one example where the summary omitted the evidence Strategy B needed to answer correctly.

> Article (excerpt): "The infrastructure engineering team successfully migrated our main cluster, achieving a stable 40% reduction in API query latency. However, due to unresolved memory leaks discovered in the new logging module, the director confirmed that the formal production release is delayed until next quarter."
>
> Summary: "The infrastructure engineering team successfully migrated the main cluster, achieving a stable 40% reduction in query latency."
>
> Question: When is the formal production release scheduled?
> Strategy B prediction: [Answer Not Found]
> Gold: next quarter
>
> What was lost in summarization: The abstractive summarization model focused heavily on the technical milestone but completely omitted the second half of the paragraph regarding the memory leaks and the delayed deployment timeline. Because this evidence was dropped during compression, my QA model had no context left to extract the correct schedule.

## 5. Recommendation

Specify quantitative thresholds for when to use each strategy. Anchor in your measured numbers.

| Use Strategy A when… | Use Strategy B when… |
|---|---|
| The context article exceeds 500 words, or when strict verbatim matching, numerical metric tracking, and high-accuracy deep-in-document lookups are required. | The document is short (under 400 words), input token constraints are extremely tight, and the application only requires general semantic or high-level summaries. |

Justification: My experimental results clearly show that Strategy A provides massive accuracy advantages over Strategy B (0.8200 F1 vs. 0.1950 F1) by preserving raw evidence. While cascading pipelines via Strategy B can save compute resources by processing fewer tokens at the QA stage, the absolute collapse of my Exact Match score down to 0.1500 proves that summarizing first causes unacceptable faithfulness loss for technical domains.