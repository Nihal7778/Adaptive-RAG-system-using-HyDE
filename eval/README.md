# Evaluation (Mini Eval Set)

This folder contains a lightweight evaluation setup to help you **prove** the chatbot works.

## Files
- `questions.jsonl` – 20–50 test questions (one JSON per line)
- `run_eval.py` – runs the eval and generates a report

## Question format (JSONL)
Each line looks like:

```json
{"id":"q01","question":"What is acne?","must_include":["acne"],"should_refuse":false}
```

Optional (for multi-doc later):

```json
{"id":"q30","question":"How do I submit electronic claims?","must_include":["claims"],"should_refuse":false,"scope":{"collection":"anthem_provider_manual_ga_2026"}}
```

## Run
From repo root:

### 1) Retrieval-only eval (no LLM cost)
```bash
python eval/run_eval.py --mode retrieval --k 3
```

### 2) Full RAG eval (calls the LLM)
```bash
python eval/run_eval.py --mode rag --k 3
```

Outputs are written to `eval_results/`.

## What to look at
- **Citation present rate** – should be near 100% once citations are wired
- **Avg keyword coverage (retrieved context)** – shows retrieval quality
- **Refusal accuracy** – for out-of-scope questions

Tip: keep 5–10 out-of-scope questions to catch hallucinations.
