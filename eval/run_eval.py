"""Simple evaluation runner for the MediRAG project.

What it measures:
- Retrieval quality: does retrieved context contain the key topic keywords?
- Refusal behavior: does the model refuse when question is out-of-scope?
- Citations presence: are we returning chunk/page sources?
- **NEW: Faithfulness: are answers grounded in retrieved context?**
- **NEW: F1 Score: token-level overlap with ground truth**
- **NEW: Answer Correctness: is the answer correct?**

Outputs:
- eval_results/<timestamp>_results.json
- eval_results/<timestamp>_results.csv
- eval_results/<timestamp>_report.md

Usage:
  python eval/run_eval.py --questions eval/questions.jsonl --mode rag
  python eval/run_eval.py --questions eval/questions.jsonl --mode retrieval

Advanced (rerank):
  python eval/run_eval.py --mode rag --rerank --k 5 --candidate_k 20
  python eval/run_eval.py --mode retrieval --rerank --k 5 --candidate_k 20

Notes:
- mode=rag calls the LLM (costs tokens).
- mode=retrieval only tests Pinecone retrieval (no LLM call).
- Supports multi-doc filtering via the optional `scope` field in questions.jsonl.
"""

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


@dataclass
class EvalCase:
    id: str
    question: str
    must_include: List[str]
    should_refuse: bool
    scope: Optional[Dict[str, Any]] = None
    ground_truth: Optional[str] = None  # NEW: for F1 calculation
    expected_pages: Optional[List[int]] = None  # NEW: for retrieval precision


REFUSAL_PATTERNS = [
    r"\bi\s+don'?t\s+know\b",
    r"\bnot\s+enough\s+information\b",
    r"\bnot\s+provided\b",
    r"\bnot\s+in\s+the\s+(document|context)\b",
    r"\bcan'?t\s+find\b",
    r"\bunable\s+to\s+answer\b",
    r"\bno\s+relevant\s+information\b",
]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def parse_questions_jsonl(path: str) -> List[EvalCase]:
    cases: List[EvalCase] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            qid = str(obj.get("id", f"q{line_no}"))
            if qid in seen_ids:
                raise ValueError(f"Duplicate id '{qid}' in {path} at line {line_no}")
            seen_ids.add(qid)

            cases.append(
                EvalCase(
                    id=qid,
                    question=str(obj["question"]),
                    must_include=list(obj.get("must_include", [])),
                    should_refuse=bool(obj.get("should_refuse", False)),
                    scope=obj.get("scope"),
                    ground_truth=obj.get("ground_truth"),  # NEW
                    expected_pages=obj.get("expected_pages"),  # NEW
                )
            )
    return cases


def build_sources(context_docs: List[Any]) -> List[str]:
    """Build human-readable sources from retrieved docs' metadata."""
    sources: List[str] = []
    seen = set()

    for d in context_docs or []:
        meta = getattr(d, "metadata", {}) or {}

        doc_name = meta.get("doc_name")
        if not doc_name:
            src = meta.get("source", "")
            doc_name = os.path.basename(src) if src else "document"

        page_display = meta.get("page_display")
        if page_display is None:
            page = meta.get("page")
            page_display = (int(page) + 1) if isinstance(page, (int, float)) else None

        chunk_id = meta.get("chunk_id")

        label = f"{doc_name}"
        if page_display is not None:
            label += f" • p.{page_display}"  # FIXED: UTF-8 bullet
        if chunk_id:
            label += f" • {chunk_id}"  # FIXED: UTF-8 bullet

        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources


def contains_refusal(answer: str) -> bool:
    text = _normalize(answer)
    return any(re.search(pat, text) for pat in REFUSAL_PATTERNS)


def keyword_hits(text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    text_n = _normalize(text)
    missing: List[str] = []
    hit = 0
    for kw in keywords:
        kw_n = _normalize(kw)
        if not kw_n:
            continue
        if kw_n in text_n:
            hit += 1
        else:
            missing.append(kw)
    return hit, missing


# ============================================
# NEW: FAITHFULNESS METRIC (MOST CRITICAL)
# ============================================

def calculate_faithfulness(question: str, answer: str, context_docs: List[Any]) -> Optional[float]:
    """
    Check if answer is grounded in retrieved context using LLM-as-judge.
    Returns score 0.0-1.0, or None if skipped.
    """
    if not answer or not context_docs:
        return None
    
    # Build context text
    context_text = "\n\n".join([
        f"[Chunk {i+1}]: {getattr(d, 'page_content', '')}" 
        for i, d in enumerate(context_docs)
    ])
    
    # LLM-as-judge prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""You are evaluating if an AI answer is faithful to retrieved medical context.

Question: {question}

Answer: {answer}

Retrieved Context:
{context_text}

Task: Verify if EVERY claim in the answer is supported by the context above.
- Paraphrasing is OK
- Adding unsupported medical facts is NOT OK
- Contradicting context is NOT OK

Rate faithfulness from 0.0 to 1.0:
1.0 = All claims supported by context
0.8 = Mostly faithful, minor unsupported detail
0.5 = Half the claims unsupported  
0.0 = Completely hallucinated or contradicts context

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.85")"""

    try:
        response = llm.invoke(prompt)
        score_text = response.content if isinstance(response.content, str) else str(response.content)
        score_text = score_text.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except Exception as e:
        print(f"  WARNING: Faithfulness check failed - {e}")
        return None


# ============================================
# NEW: F1 SCORE METRIC
# ============================================

def calculate_f1_score(prediction: str, ground_truth: str) -> Optional[float]:
    """
    Token-level F1 score between prediction and ground truth.
    Standard metric for QA evaluation (used in SQuAD, BioASQ).
    Returns score 0.0-1.0, or None if ground_truth not provided.
    """
    if not ground_truth or not prediction:
        return None
    
    # Normalize and tokenize
    def normalize_answer(s):
        """Lower text, remove punctuation, extra whitespace."""
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s)  # Remove punctuation
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    # Count common tokens
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    
    common_tokens = pred_counter & truth_counter
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    # F1 = harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


# ============================================
# NEW: Accuracy METRIC
# ============================================

def calculate_answer_correctness(question: str, answer: str, ground_truth: str) -> Optional[bool]:
    """
    Binary correctness: Is the answer medically correct?
    Uses LLM-as-judge for evaluation.
    Returns True/False or None if no ground truth.
    """
    if not ground_truth or not answer:
        return None
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""You are a medical expert evaluating answercorrectness.

Question: {question}

Ground Truth Answer: {ground_truth}

AI Answer: {answer}

Task: Is the AI answer medically CORRECT and accurately addresses the question?
- Minor wording differences are OK
- Must contain the key medical facts
- Must not have major errors or omissions

Respond with ONLY: "CORRECT" or "INCORRECT"
"""

    try:
        response = llm.invoke(prompt)
        result = response.content.strip().upper()
        return result == "CORRECT"
    except Exception as e:
        print(f"  WARNING: Correctness check failed - {e}")
        return None


# -----------------------------
# Advanced RAG: Reranker wrapper
# -----------------------------
class RerankRetriever:
    """
    Wrap a base retriever:
      - retrieve candidate_k docs
      - rerank with a cross-encoder
      - return top_n docs
    Compatible with LangChain chains (invoke + get_relevant_documents).
    """

    def __init__(self, base_retriever: Any, cross_encoder: Any, top_n: int, max_chars: int = 2000):
        self.base = base_retriever
        self.ce = cross_encoder
        self.top_n = top_n
        self.max_chars = max_chars

     # LangChain Runnable compatibility (create_retrieval_chain expects this)
    def with_config(self, *args, **kwargs):
        new_base = self.base
        if hasattr(self.base, "with_config"):
            try:
                new_base = self.base.with_config(*args, **kwargs)
            except Exception:
                # If signature mismatch, just keep base as-is
                new_base = self.base

        return RerankRetriever(
            base_retriever=new_base,
            cross_encoder=self.ce,
            top_n=self.top_n,
            max_chars=self.max_chars,
        )

    # Optional: forward unknown attributes to the wrapped retriever (prevents more missing-method errors)
    def __getattr__(self, name: str):
        return getattr(self.base, name)


    def invoke(self, query: str, config: Optional[dict] = None):
        docs = self.base.invoke(query)
        if not docs:
            return docs

        pairs = [(query, (getattr(d, "page_content", "") or "")[: self.max_chars]) for d in docs]
        scores = self.ce.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[: self.top_n]]

    # Some LC versions/chains call this
    def get_relevant_documents(self, query: str):
        return self.invoke(query)



def build_shared_components(index_name: str):
    """Build components once; we will create a filtered retriever per test case."""
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_completion_tokens=800)
    doc_chain = create_stuff_documents_chain(llm, prompt)

    return docsearch, doc_chain


def make_retriever(
    docsearch: PineconeVectorStore,
    k: int,
    scope: Optional[Dict[str, Any]],
    rerank: bool = False,
    candidate_k: int = 20,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    """Create a retriever. If scope exists, apply it as a Pinecone metadata filter.
    If rerank=True: retrieve candidate_k docs then rerank down to k."""
    search_k = candidate_k if rerank else k

    kwargs: Dict[str, Any] = {"k": search_k}
    if scope:
        kwargs["filter"] = dict(scope)

    base = docsearch.as_retriever(search_type="similarity", search_kwargs=kwargs)

    if not rerank:
        return base

    from sentence_transformers import CrossEncoder  # will raise if not installed

    ce = CrossEncoder(rerank_model)
    return RerankRetriever(base_retriever=base, cross_encoder=ce, top_n=k)



def run_eval(
    index_name: str,
    questions_path: str,
    mode: str,
    k: int,
    out_dir: str,
    rerank: bool,
    candidate_k: int,
    rerank_model: str,
) -> str:

    cases = parse_questions_jsonl(questions_path)
    docsearch, doc_chain = build_shared_components(index_name=index_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"{ts}")
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for case in cases:
        retriever = make_retriever(
         docsearch,
         k=k,
         scope=case.scope,
            rerank=rerank,
            candidate_k=candidate_k,
            rerank_model=rerank_model,
)


        t0 = time.time()
        context_docs = retriever.invoke(case.question)

        answer = ""
        if mode == "rag":
            # We already retrieved (and reranked) docs above
            # Now directly run the doc chain using the retrieved docs as context
            resp = doc_chain.invoke({"input": case.question, "context": context_docs})

            # doc_chain sometimes returns a string, sometimes a dict depending on versions
            if isinstance(resp, dict):
                answer = resp.get("answer") or resp.get("output_text") or resp.get("text") or ""
            else:
                answer = str(resp)


        dt_ms = int((time.time() - t0) * 1000)

        sources = build_sources(context_docs)
        context_text = "\n\n".join([getattr(d, "page_content", "") for d in context_docs or []])

        # === EXISTING SCORING ===
        refusal_pred = contains_refusal(answer) if answer else False
        refusal_ok = (refusal_pred == case.should_refuse) if mode == "rag" else None

        ans_hit, ans_missing = keyword_hits(answer, case.must_include) if answer else (0, list(case.must_include))
        ctx_hit, ctx_missing = keyword_hits(context_text, case.must_include) if context_text else (0, list(case.must_include))

        # === NEW: FAITHFULNESS (only for non-refusal medical questions) ===
        faithfulness = None
        if mode == "rag" and answer and not case.should_refuse:
            faithfulness = calculate_faithfulness(case.question, answer, context_docs)
            if faithfulness is not None:
                print(f"  [{case.id}] Faithfulness: {faithfulness:.2f}")
        
        # === NEW: F1 SCORE (only if ground truth provided) ===
        f1_score = None
        if mode == "rag" and answer and case.ground_truth and not case.should_refuse:
            f1_score = calculate_f1_score(answer, case.ground_truth)
            if f1_score is not None:
                print(f"  [{case.id}] F1 Score: {f1_score:.2f}")
        
        # === NEW: RETRIEVAL PRECISION (only if expected_pages provided) ===
        retrieval_metrics = None
        if case.expected_pages:
            answer_correctness = calculate_answer_correctness(
            case.question, 
            answer, 
            case.ground_truth
            ) if case.ground_truth and not case.should_refuse else None
            if retrieval_metrics:
                print(f"  [{case.id}] Found correct page: {retrieval_metrics['found_correct_page']}")

        results.append(
            {
                "id": case.id,
                "question": case.question,
                "mode": mode,
                "k": k,
                "scope": case.scope,
                "should_refuse": case.should_refuse,
                "answer": answer,
                "latency_ms": dt_ms,
                "retrieved_chunks": len(context_docs or []),
                "citations_present": len(sources) > 0,
                "sources": sources,
                "must_include": case.must_include,
                "answer_keyword_hits": ans_hit,
                "answer_keyword_missing": ans_missing,
                "context_keyword_hits": ctx_hit,
                "context_keyword_missing": ctx_missing,
                "refusal_ok": refusal_ok,
                # NEW FIELDS:
                "faithfulness": faithfulness,
                "f1_score": f1_score,
                "answer_correctness": answer_correctness,
            }
        )

    # write outputs
    json_path = base + "_results.json"
    csv_path = base + "_results.csv"
    md_path = base + "_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "id",
        "question",
        "mode",
        "k",
        "should_refuse",
        "latency_ms",
        "retrieved_chunks",
        "citations_present",
        "context_keyword_hits",
        "answer_keyword_hits",
        "refusal_ok",
        "faithfulness",  # NEW
        "f1_score",  # NEW
        "answer_correctness",  # NEW
        "answer",
        "sources",
        "scope",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            row["sources"] = "; ".join(r.get("sources", []))
            w.writerow(row)

    total = len(results)
    cite_rate = sum(1 for r in results if r.get("citations_present")) / max(total, 1)
    ctx_cov = sum((r.get("context_keyword_hits", 0) / max(len(r.get("must_include", [])), 1)) for r in results) / max(total, 1)
    ans_cov = sum((r.get("answer_keyword_hits", 0) / max(len(r.get("must_include", [])), 1)) for r in results) / max(total, 1)

    refusal_rows = [r for r in results if r.get("refusal_ok") is not None]
    refusal_acc = (sum(1 for r in refusal_rows if r.get("refusal_ok")) / max(len(refusal_rows), 1)) if refusal_rows else None

    # NEW: Calculate aggregate metrics for new scores
    faithfulness_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    
    f1_scores = [r["f1_score"] for r in results if r.get("f1_score") is not None]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
    
    correctness_scores = [r["answer_correctness"] for r in results if r.get("answer_correctness") is not None]
    total_accuracy = (sum(correctness_scores) / len(correctness_scores) * 100) if correctness_scores else None
    
    correct_page_hits = [r["found_correct_page"] for r in results if r.get("found_correct_page") is not None]
    page_hit_rate = sum(correct_page_hits) / len(correct_page_hits) if correct_page_hits else None

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# MediRAG Eval Report ({ts})\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Mode: **{mode}**\n")
        f.write(f"- Top-k: **{k}**\n")
        f.write(f"- Total questions: **{total}**\n\n")
        
        f.write(f"## Core Metrics\n")
        f.write(f"- Citation present rate: **{cite_rate:.0%}**\n")
        f.write(f"- Avg keyword coverage (retrieved context): **{ctx_cov:.0%}**\n")
        if mode == "rag":
            f.write(f"- Avg keyword coverage (answer): **{ans_cov:.0%}**\n")
        if refusal_acc is not None:
            f.write(f"- Refusal accuracy: **{refusal_acc:.0%}**\n")
        
        # NEW METRICS IN REPORT:
        if avg_faithfulness is not None or avg_f1 is not None or total_accuracy is not None:
            f.write(f"\n## Advanced Metrics (NEW)\n")
            
        if avg_faithfulness is not None:
            f.write(f"- **Avg Faithfulness (Groundedness): {avg_faithfulness:.0%}**  CRITICAL\n")
        if avg_f1 is not None:
            f.write(f"- **Avg F1 Score (vs ground truth): {avg_f1:.0%}** \n")
        if total_accuracy is not None:
             f.write(f"- **Total Accuracy: {total_accuracy:.0%}** (Correct answers)\n")
        if page_hit_rate is not None:
            f.write(f"- **Page Hit Rate: {page_hit_rate:.0%}**\n")
        
        f.write("\n---\n\n")
        f.write("## Examples (first 10)\n\n")

        for r in results[:10]:
            f.write(f"### {r['id']}\n")
            f.write(f"**Q:** {r['question']}\n\n")
            if r.get("answer"):
                f.write(f"**A:** {r['answer']}\n\n")
            
            # Show new metrics in examples
            if r.get("faithfulness") is not None:
                f.write(f"**Faithfulness:** {r['faithfulness']:.0%}\n\n")
            if r.get("f1_score") is not None:
                f.write(f"**F1 Score:** {r['f1_score']:.0%}\n\n")
            
            if r.get("sources"):
                f.write("**Sources:**\n")
                for s in r["sources"][:5]:
                    f.write(f"- {s}\n")
                f.write("\n")
            if r.get("must_include"):
                f.write(f"**Must include:** {', '.join(r['must_include'])}\n\n")
            f.write("---\n\n")

    print(f"\n✅ Evaluation complete!")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    return md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="medicalbot", help="Pinecone index name")
    parser.add_argument("--questions", default="eval/questions.jsonl", help="Path to JSONL questions")
    parser.add_argument("--mode", choices=["rag", "retrieval"], default="rag")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default="eval_results")
    # Advanced RAG knobs
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--candidate_k", type=int, default=20, help="Docs to fetch BEFORE reranking (must be >= k)")
    parser.add_argument(
        "--rerank_model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="SentenceTransformers CrossEncoder model name",
    )


    args = parser.parse_args()
    if args.rerank and args.candidate_k < args.k:
        raise ValueError("--candidate_k must be >= --k when --rerank is enabled")


    run_eval(
        index_name=args.index,
        questions_path=args.questions,
        mode=args.mode,
        k=args.k,
        out_dir=args.out,
        rerank=args.rerank,
        candidate_k=args.candidate_k,
        rerank_model=args.rerank_model,
    )


if __name__ == "__main__":
    main()