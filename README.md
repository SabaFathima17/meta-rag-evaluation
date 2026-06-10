# Meta-RAG: Using Retrieval to Evaluate Retrieval

A reference-free evaluation framework for RAG systems that uses a second
retrieval layer to assess whether generated answers are specifically grounded
in their retrieved context, without requiring gold labels.

## Research Question

Can a second retrieval pass detect when a RAG answer is poorly grounded
in its source, using only the question, the retrieved passage, and the answer
itself, with no access to human-written gold labels?

## The Core Idea: Specificity Ratio

Standard faithfulness checks ask: does the answer text appear in the context?
For extractive systems this is always yes, making it uninformative.

This project introduces the **specificity ratio**:

1. Run standard RAG: retrieve a passage, extract an answer
2. Use the answer as a new query to retrieve an independent evidence passage
3. Measure how much the answer overlaps with the original retrieved context
4. Measure how much the answer overlaps with the independent evidence passage
5. Compute the ratio: original overlap divided by independent overlap

A high ratio means the answer is specific to its source.
A low ratio means the answer is generic enough to fit any passage.

This distinction is impossible to make with n-gram faithfulness alone.

## Results (n=200 answerable questions, seed=42)

| Verdict | Count | Percentage | Mean F1 | Retrieval Correct |
|---|---|---|---|---|
| SUPPORTED | 188 | 94.0% | 0.163 | 81.4% |
| PARTIAL | 12 | 6.0% | 0.128 | 66.7% |
| UNSUPPORTED | 0 | 0.0% | n/a | n/a |

Key finding: PARTIAL cases have 14.7 percentage points lower retrieval
accuracy than SUPPORTED cases. The meta-evaluator catches genuinely weaker
answers without ever accessing gold labels.

## Why This Matters

Standard evaluation requires gold labels. For every question you need a
human-written correct answer to compare against. This is expensive and
does not scale to new domains.

Reference-free evaluation using the specificity ratio works without gold labels.
It can be applied to any domain where you have a corpus of passages and a
retrieval system, even when you have no human annotations.

This approach becomes significantly more powerful for generative RAG systems
(LLM-based) where the model can produce text not present in the retrieved
passage. In that setting the specificity ratio would catch hallucinated claims
that are generic rather than grounded in the specific source.

## Connection to Perplexity Research

Perplexity's DRACO benchmark (February 2026) evaluates AI answers on
citation quality as one of its four core axes. Citation quality asks exactly
this: is each claim in the answer actually supported by the cited source?

The specificity ratio is a lightweight, reference-free approximation of this.
It does not require an LLM judge or human rubric. It uses retrieval itself
to check retrieval outputs.

## Project Structure

```
meta-rag-eval/
|
|-- meta_rag_eval.ipynb     main analysis notebook
|-- README.md               this file
|-- requirements.txt        dependencies
|
|-- outputs/
    |-- plot_01_verdict_distribution.png
    |-- plot_02_specificity_distribution.png
    |-- plot_03_verdict_vs_retrieval.png
    |-- plot_04_f1_by_verdict.png
    |-- plot_05_threshold_sensitivity.png
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook meta_rag_eval.ipynb
```

Place SQuAD.json in the same folder. Run all cells top to bottom.

## Limitations

Extractive systems inflate specificity scores because copied sentences
always overlap with their source. This limitation is documented in the
notebook and is the motivation for applying this approach to generative
systems where the limitation does not hold.

The specificity ratio is a heuristic signal, not a ground truth measurement.
It should be treated as one signal among several, not a definitive verdict.

## Tech Stack

Python, rank-bm25, scikit-learn, pandas, NumPy, Matplotlib, Seaborn
