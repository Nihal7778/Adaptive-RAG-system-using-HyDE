"""
Comparative evaluation: Baseline RAG vs HyDE RAG
Runs same questions through both systems and compares metrics.
"""

import json
import time
from datetime import datetime
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from collections import Counter
import re

from src.helper import download_hugging_face_embeddings
from src.hyde import HyDERetriever
from src.prompt import system_prompt

load_dotenv()

# Initialize components
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="medicalbot"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000)
hyde_retriever = HyDERetriever()


def load_test_questions(filepath="eval/questions.jsonl"):
    """Load test questions"""
    questions = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score"""
    if not ground_truth or not prediction:
        return None
    
    def normalize(s):
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    common = pred_counter & truth_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)


def calculate_faithfulness(question: str, answer: str, context_docs: List) -> float:
    """Check if answer is grounded in context"""
    if not answer or not context_docs:
        return None
    
    context_text = "\n\n".join([
        f"[Chunk {i+1}]: {doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])
    
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Rate faithfulness (0.0-1.0) of this answer to the retrieved context.

Question: {question}
Answer: {answer}
Context: {context_text}

1.0 = All claims supported
0.5 = Half supported
0.0 = Hallucinated

Respond with only a number:"""
    
    try:
        response = judge_llm.invoke(prompt)
        return float(response.content.strip())
    except:
        return 0.5


def calculate_retrieval_precision(context_docs: List, expected_pages: List[int]) -> float:
    """Calculate retrieval precision"""
    if not expected_pages or not context_docs:
        return None
    
    retrieved_pages = set()
    for doc in context_docs:
        meta = getattr(doc, "metadata", {}) or {}
        page = meta.get("page_display") or meta.get("page")
        if page:
            retrieved_pages.add(int(page))
    
    expected_set = set(expected_pages)
    relevant = retrieved_pages & expected_set
    
    return len(relevant) / len(context_docs) if context_docs else 0.0


def run_baseline(question: str, k: int = 5):
    """Run baseline RAG (direct query embedding)"""
    start_time = time.time()
    
    # Retrieve using question directly
    retrieved_docs = docsearch.similarity_search(question, k=k)
    
    # Generate answer
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt_text = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
    
    answer_response = llm.invoke(prompt_text)
    answer = answer_response.content
    
    total_time = time.time() - start_time
    
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "latency": total_time,
        "hypothesis": None  # Baseline doesn't use hypothesis
    }


def run_hyde(question: str, k: int = 5):
    """Run HyDE RAG (hypothetical document embedding)"""
    start_time = time.time()
    
    # Use HyDE to retrieve
    retrieved_docs, hypothesis, gen_time = hyde_retriever.retrieve_with_hyde(
        docsearch, question, k=k
    )
    
    # Generate answer
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt_text = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
    
    answer_response = llm.invoke(prompt_text)
    answer = answer_response.content
    
    total_time = time.time() - start_time
    
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "latency": total_time,
        "hypothesis": hypothesis,
        "hypothesis_generation_time": gen_time
    }


def compare_systems(test_questions: List[Dict], k: int = 5):
    """Run comparative evaluation"""
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(test_questions),
            "k": k
        },
        "questions": []
    }
    
    print(f"\n{'='*60}")
    print(f"COMPARATIVE EVALUATION: Baseline vs HyDE")
    print(f"{'='*60}\n")
    print(f"Testing on {len(test_questions)} questions...\n")
    
    for i, q in enumerate(test_questions, 1):
        if q.get("should_refuse"):
            continue  # Skip non-medical questions
        
        question_id = q.get("id")
        question_text = q.get("question")
        ground_truth = q.get("ground_truth")
        expected_pages = q.get("expected_pages")
        
        print(f"[{i}/{len(test_questions)}] {question_text[:50]}...")
        
        # Run baseline
        print("  Running baseline...", end=" ")
        baseline_result = run_baseline(question_text, k=k)
        print(f"✓ ({baseline_result['latency']:.2f}s)")
        
        # Run HyDE
        print("  Running HyDE...", end=" ")
        hyde_result = run_hyde(question_text, k=k)
        print(f"✓ ({hyde_result['latency']:.2f}s)")
        
        # Calculate metrics for both
        baseline_metrics = {}
        hyde_metrics = {}
        
        # Faithfulness
        baseline_metrics['faithfulness'] = calculate_faithfulness(
            question_text, baseline_result['answer'], baseline_result['retrieved_docs']
        )
        hyde_metrics['faithfulness'] = calculate_faithfulness(
            question_text, hyde_result['answer'], hyde_result['retrieved_docs']
        )
        
        # F1 Score (if ground truth available)
        if ground_truth:
            baseline_metrics['f1_score'] = calculate_f1_score(
                baseline_result['answer'], ground_truth
            )
            hyde_metrics['f1_score'] = calculate_f1_score(
                hyde_result['answer'], ground_truth
            )
        
        # Retrieval Precision (if expected pages available)
        if expected_pages:
            baseline_metrics['retrieval_precision'] = calculate_retrieval_precision(
                baseline_result['retrieved_docs'], expected_pages
            )
            hyde_metrics['retrieval_precision'] = calculate_retrieval_precision(
                hyde_result['retrieved_docs'], expected_pages
            )
        
        # Store results
        question_result = {
            "id": question_id,
            "question": question_text,
            "baseline": {
                "answer": baseline_result['answer'],
                "latency": baseline_result['latency'],
                **baseline_metrics
            },
            "hyde": {
                "answer": hyde_result['answer'],
                "hypothesis": hyde_result['hypothesis'],
                "latency": hyde_result['latency'],
                "hypothesis_time": hyde_result['hypothesis_generation_time'],
                **hyde_metrics
            }
        }
        
        results["questions"].append(question_result)
        
        # Show comparison
        print(f"    Baseline: Faith={baseline_metrics.get('faithfulness', 0):.2f}, "
              f"F1={baseline_metrics.get('f1_score', 0):.2f}, "
              f"Precision={baseline_metrics.get('retrieval_precision', 0):.2f}")
        print(f"    HyDE:     Faith={hyde_metrics.get('faithfulness', 0):.2f}, "
              f"F1={hyde_metrics.get('f1_score', 0):.2f}, "
              f"Precision={hyde_metrics.get('retrieval_precision', 0):.2f}")
        print()
    
    # Calculate aggregates
    aggregate = calculate_aggregates(results["questions"])
    results["aggregate"] = aggregate
    
    return results


def calculate_aggregates(question_results: List[Dict]) -> Dict:
    """Calculate aggregate metrics"""
    
    baseline_faith = [q["baseline"]["faithfulness"] for q in question_results 
                      if q["baseline"].get("faithfulness") is not None]
    hyde_faith = [q["hyde"]["faithfulness"] for q in question_results 
                  if q["hyde"].get("faithfulness") is not None]
    
    baseline_f1 = [q["baseline"]["f1_score"] for q in question_results 
                   if q["baseline"].get("f1_score") is not None]
    hyde_f1 = [q["hyde"]["f1_score"] for q in question_results 
               if q["hyde"].get("f1_score") is not None]
    
    baseline_precision = [q["baseline"]["retrieval_precision"] for q in question_results 
                          if q["baseline"].get("retrieval_precision") is not None]
    hyde_precision = [q["hyde"]["retrieval_precision"] for q in question_results 
                      if q["hyde"].get("retrieval_precision") is not None]
    
    baseline_latency = [q["baseline"]["latency"] for q in question_results]
    hyde_latency = [q["hyde"]["latency"] for q in question_results]
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    return {
        "baseline": {
            "avg_faithfulness": avg(baseline_faith),
            "avg_f1_score": avg(baseline_f1),
            "avg_retrieval_precision": avg(baseline_precision),
            "avg_latency": avg(baseline_latency)
        },
        "hyde": {
            "avg_faithfulness": avg(hyde_faith),
            "avg_f1_score": avg(hyde_f1),
            "avg_retrieval_precision": avg(hyde_precision),
            "avg_latency": avg(hyde_latency)
        },
        "improvement": {
            "faithfulness": ((avg(hyde_faith) - avg(baseline_faith)) / avg(baseline_faith) * 100) if baseline_faith else 0,
            "f1_score": ((avg(hyde_f1) - avg(baseline_f1)) / avg(baseline_f1) * 100) if baseline_f1 else 0,
            "retrieval_precision": ((avg(hyde_precision) - avg(baseline_precision)) / avg(baseline_precision) * 100) if baseline_precision else 0,
            "latency_overhead": avg(hyde_latency) - avg(baseline_latency)
        }
    }


def generate_report(results: Dict, output_file: str):
    """Generate comparison report"""
    
    agg = results["aggregate"]
    
    report = f"""# HyDE vs Baseline RAG - Comparison Report

**Generated:** {results['metadata']['timestamp']}
**Questions Tested:** {results['metadata']['total_questions']}
**Retrieval K:** {results['metadata']['k']}

---

## Aggregate Metrics

| Metric | Baseline | HyDE | Improvement |
|--------|----------|------|-------------|
| **Faithfulness** | {agg['baseline']['avg_faithfulness']:.1%} | {agg['hyde']['avg_faithfulness']:.1%} | **{agg['improvement']['faithfulness']:+.1f}%** |
| **F1 Score** | {agg['baseline']['avg_f1_score']:.1%} | {agg['hyde']['avg_f1_score']:.1%} | **{agg['improvement']['f1_score']:+.1f}%** |
| **Retrieval Precision** | {agg['baseline']['avg_retrieval_precision']:.1%} | {agg['hyde']['avg_retrieval_precision']:.1%} | **{agg['improvement']['retrieval_precision']:+.1f}%** |
| **Latency** | {agg['baseline']['avg_latency']:.2f}s | {agg['hyde']['avg_latency']:.2f}s | +{agg['improvement']['latency_overhead']:.2f}s |

---

## Key Findings

### Strengths of HyDE:
- {"✅ Better retrieval precision" if agg['improvement']['retrieval_precision'] > 5 else "⚠️ Similar retrieval precision"}
- {"✅ Higher answer quality (F1)" if agg['improvement']['f1_score'] > 5 else "⚠️ Similar answer quality"}
- {"✅ Improved faithfulness" if agg['improvement']['faithfulness'] > 2 else "⚠️ Similar faithfulness"}

### Trade-offs:
- ⏱️ Adds ~{agg['improvement']['latency_overhead']:.2f}s latency per query
- 💰 Uses 2x LLM calls (higher cost)
- 🔧 More complex implementation

---

## Detailed Question-by-Question Results

"""
    
    for q_result in results["questions"][:10]:  # First 10 questions
        report += f"""### {q_result['id']}: {q_result['question']}

**Baseline:**
- Answer: {q_result['baseline']['answer'][:200]}...
- Metrics: Faith={q_result['baseline'].get('faithfulness', 0):.2f}, F1={q_result['baseline'].get('f1_score', 0):.2f}

**HyDE:**
- Hypothesis: "{q_result['hyde']['hypothesis'][:150]}..."
- Answer: {q_result['hyde']['answer'][:200]}...
- Metrics: Faith={q_result['hyde'].get('faithfulness', 0):.2f}, F1={q_result['hyde'].get('f1_score', 0):.2f}

---

"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {output_file}")


if __name__ == "__main__":
    # Load questions
    questions = load_test_questions()
    
    # Run comparison
    results = compare_systems(questions, k=5)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"eval_results/{timestamp}_hyde_comparison.json"
    report_file = f"eval_results/{timestamp}_hyde_comparison.md"
    
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    generate_report(results, report_file)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    agg = results["aggregate"]
    print(f"Baseline Faithfulness: {agg['baseline']['avg_faithfulness']:.1%}")
    print(f"HyDE Faithfulness:     {agg['hyde']['avg_faithfulness']:.1%}")
    print(f"Improvement:           {agg['improvement']['faithfulness']:+.1f}%")
    print()
    print(f"Baseline F1:           {agg['baseline']['avg_f1_score']:.1%}")
    print(f"HyDE F1:               {agg['hyde']['avg_f1_score']:.1%}")
    print(f"Improvement:           {agg['improvement']['f1_score']:+.1f}%")
    print()
    print(f"Baseline Precision:    {agg['baseline']['avg_retrieval_precision']:.1%}")
    print(f"HyDE Precision:        {agg['hyde']['avg_retrieval_precision']:.1%}")
    print(f"Improvement:           {agg['improvement']['retrieval_precision']:+.1f}%")
    print()
    print(f"Latency Overhead:      +{agg['improvement']['latency_overhead']:.2f}s")
    print(f"{'='*60}\n")
