"""
NITP RAG System - Quantitative Evaluation Benchmark
Evaluates Pure Vector, Pure BM25, and Hybrid RRF on a 30-question ground-truth test suite.
Computes: Hit Rate@K (K=1, 5, 10, 20), Mean Reciprocal Rank (MRR), and Zero-Hallucination Rate.
"""
import os
import json
import time
import numpy as np
from typing import List, Dict, Any

from src.embeddings.generator import GeminiEmbedder
from src.storage.vector_store import ChromaVectorStore
from src.generation.llm import GeminiLLM

DATASET_PATH = "evaluation/dataset.json"
RESULTS_MD_PATH = "evaluation/results.md"

def load_dataset() -> List[Dict[str, Any]]:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Evaluation dataset not found at {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def is_ground_truth_match(doc_text: str, metadata: dict, target_source: str, target_keywords: List[str]) -> bool:
    """Checks if a retrieved chunk matches both target source file and mandatory semantic keywords."""
    if target_source and metadata.get("source") != target_source:
        return False
    doc_lower = doc_text.lower()
    for kw in target_keywords:
        if kw.lower() not in doc_lower:
            return False
    return True

def run_evaluation(test_llm_traps: bool = True):
    print("================================================================")
    print("      NIT PATNA RAG SYSTEM - QUANTITATIVE EVALUATION BENCHMARK   ")
    print("================================================================\n")
    
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} evaluation questions from {DATASET_PATH}")
    
    factual_cases = [c for c in dataset if not c.get("is_trap", False)]
    trap_cases = [c for c in dataset if c.get("is_trap", False)]
    print(f"Factual queries: {len(factual_cases)} | Trap guardrail queries: {len(trap_cases)}\n")
    
    print("Initializing RAG pipeline and ChromaDB...")
    embedder = GeminiEmbedder()
    db = ChromaVectorStore()
    llm = GeminiLLM() if test_llm_traps else None
    
    # Initialize BM25 if not already lazy-loaded
    _dummy_vec = embedder.embed_text("test")
    db.hybrid_search("test", _dummy_vec, n_results=1)
    
    # Tracking metrics for the three engines
    methods = ["Dense Vector", "Sparse BM25", "Hybrid RRF (Proposed)"]
    metrics = {
        m: {
            "ranks": [],
            "hits_1": 0,
            "hits_5": 0,
            "hits_10": 0,
            "hits_20": 0,
            "mrr_sum": 0.0,
            "latencies": []
        }
        for m in methods
    }
    
    K_MAX = 25
    
    print("\n--- Evaluating Retrieval Performance across Factual Queries ---")
    for idx, item in enumerate(factual_cases, 1):
        q_id = item["id"]
        query = item["query"]
        target_src = item["target_source"]
        target_kw = item["target_keywords"]
        
        # 1. Embed query once
        t0 = time.time()
        q_vec = embedder.embed_text(query)
        embed_time = time.time() - t0
        
        # --- A. Pure Dense Vector Search ---
        t_dense = time.time()
        dense_res = db.collection.query(
            query_embeddings=[q_vec],
            n_results=K_MAX
        )
        t_dense_elapsed = time.time() - t_dense + embed_time
        dense_docs = dense_res['documents'][0]
        dense_metas = dense_res['metadatas'][0]
        
        dense_rank = None
        for r, (doc, meta) in enumerate(zip(dense_docs, dense_metas), 1):
            if is_ground_truth_match(doc, meta, target_src, target_kw):
                dense_rank = r
                break
                
        # --- B. Pure Sparse BM25 Search ---
        t_bm25 = time.time()
        tok_query = db.tokenize_fn(query, is_query=True)
        bm25_scores = db.bm25.get_scores(tok_query)
        top_indices = np.argsort(bm25_scores)[::-1][:K_MAX]
        t_bm25_elapsed = time.time() - t_bm25
        
        bm25_rank = None
        for r, doc_idx in enumerate(top_indices, 1):
            doc = db.all_docs[doc_idx]
            meta = db.all_metadatas[doc_idx]
            if is_ground_truth_match(doc, meta, target_src, target_kw):
                bm25_rank = r
                break
                
        # --- C. Hybrid Search (Vector + BM25 + RRF) ---
        t_hybrid = time.time()
        hybrid_res = db.hybrid_search(query_text=query, query_embedding=q_vec, n_results=K_MAX)
        t_hybrid_elapsed = time.time() - t_hybrid + embed_time
        hybrid_docs = hybrid_res['documents'][0]
        hybrid_metas = hybrid_res['metadatas'][0]
        
        hybrid_rank = None
        for r, (doc, meta) in enumerate(zip(hybrid_docs, hybrid_metas), 1):
            if is_ground_truth_match(doc, meta, target_src, target_kw):
                hybrid_rank = r
                break
                
        # Update metrics
        results_map = {
            "Dense Vector": (dense_rank, t_dense_elapsed),
            "Sparse BM25": (bm25_rank, t_bm25_elapsed),
            "Hybrid RRF (Proposed)": (hybrid_rank, t_hybrid_elapsed)
        }
        
        for m, (rnk, lat) in results_map.items():
            metrics[m]["latencies"].append(lat)
            if rnk is not None:
                metrics[m]["ranks"].append(rnk)
                if rnk <= 1:
                    metrics[m]["hits_1"] += 1
                if rnk <= 5:
                    metrics[m]["hits_5"] += 1
                if rnk <= 10:
                    metrics[m]["hits_10"] += 1
                if rnk <= 20:
                    metrics[m]["hits_20"] += 1
                metrics[m]["mrr_sum"] += (1.0 / rnk)
            else:
                metrics[m]["ranks"].append(999)
                
        print(f"[{q_id}] \"{query[:45]}...\" -> Vector: #{dense_rank or '>25'} | BM25: #{bm25_rank or '>25'} | Hybrid: #{hybrid_rank or '>25'}")

    # --- Evaluate Negative Guardrails (Trap Queries) ---
    trap_refusals = 0
    if test_llm_traps and llm:
        print("\n--- Evaluating Zero-Hallucination Guardrail on Trap Queries ---")
        for item in trap_cases:
            q_id = item["id"]
            query = item["query"]
            q_vec = embedder.embed_text(query)
            res = db.hybrid_search(query_text=query, query_embedding=q_vec, n_results=15)
            context = "\n\n".join(res['documents'][0])
            ans = llm.generate_answer(query=query, context=context)
            refused = "I don't know" in ans or "not contained" in ans.lower() or "not mentioned" in ans.lower()
            if refused:
                trap_refusals += 1
            status_str = "PASSED (Refused Hallucination)" if refused else "FAILED (Hallucinated)"
            print(f"[{q_id}] \"{query[:45]}...\" -> {status_str}")

    num_factual = len(factual_cases)
    
    # Calculate final percentages
    summary = {}
    for m in methods:
        h1 = (metrics[m]["hits_1"] / num_factual) * 100
        h5 = (metrics[m]["hits_5"] / num_factual) * 100
        h10 = (metrics[m]["hits_10"] / num_factual) * 100
        h20 = (metrics[m]["hits_20"] / num_factual) * 100
        mrr = metrics[m]["mrr_sum"] / num_factual
        avg_lat = np.mean(metrics[m]["latencies"]) * 1000 # in ms
        summary[m] = {
            "Hit@1": h1,
            "Hit@5": h5,
            "Hit@10": h10,
            "Hit@20": h20,
            "MRR": mrr,
            "Avg_Latency_ms": avg_lat
        }
        
    trap_accuracy = (trap_refusals / len(trap_cases) * 100) if trap_cases else 100.0
    
    # Print Terminal Report
    print("\n" + "=" * 80)
    print("                     EMPIRICAL BENCHMARK RESULTS TABLE")
    print("=" * 80)
    print(f"{'Retrieval Engine':<26} | {'Hit@1 (%)':<9} | {'Hit@5 (%)':<9} | {'Hit@10 (%)':<10} | {'Hit@20 (%)':<10} | {'MRR':<7} | {'Latency':<8}")
    print("-" * 80)
    for m in methods:
        s = summary[m]
        print(f"{m:<26} | {s['Hit@1']:>8.1f}% | {s['Hit@5']:>8.1f}% | {s['Hit@10']:>9.1f}% | {s['Hit@20']:>9.1f}% | {s['MRR']:>6.3f} | {s['Avg_Latency_ms']:>6.1f}ms")
    print("-" * 80)
    print(f"Zero-Hallucination Guardrail Rejection Rate: {trap_accuracy:.1f}% ({trap_refusals}/{len(trap_cases)} traps correctly refused)")
    print("=" * 80)
    
    # Generate Markdown and LaTeX Report
    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Quantitative Evaluation Benchmark Results\n\n")
        f.write(f"**Test Suite Size:** {len(dataset)} queries ({num_factual} factual benchmark queries + {len(trap_cases)} negative guardrail trap queries)\n")
        f.write(f"**Indexed Documents:** 10 NIT Patna official circulars (76 chunks)\n")
        f.write(f"**Embedding Model:** `gemini-embedding-2` (3072-dimensional vectors)\n")
        f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Empirical Retrieval Performance\n\n")
        f.write("| Retrieval Engine | Hit@1 (%) | Hit@5 (%) | Hit@10 (%) | Hit@20 (%) | MRR | Latency (ms) |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for m in methods:
            s = summary[m]
            f.write(f"| **{m}** | {s['Hit@1']:.1f}% | {s['Hit@5']:.1f}% | {s['Hit@10']:.1f}% | {s['Hit@20']:.1f}% | {s['MRR']:.3f} | {s['Avg_Latency_ms']:.1f}ms |\n")
            
        f.write(f"\n**Zero-Hallucination Guardrail Rejection Rate:** `{trap_accuracy:.1f}%` ({trap_refusals}/{len(trap_cases)} out-of-scope traps correctly refused).\n\n")
        
        f.write("## 2. LaTeX Table Snippet for Thesis & Slide 15\n\n")
        f.write("```latex\n")
        f.write("\\begin{table}[h]\n")
        f.write("    \\centering\n")
        f.write("    \\caption{Empirical Retrieval Comparison on NIT Patna Institutional Dataset}\n")
        f.write("    \\begin{tabular}{lcccccc}\n")
        f.write("        \\toprule\n")
        f.write("        \\textbf{Retrieval Strategy} & \\textbf{Hit@1} & \\textbf{Hit@5} & \\textbf{Hit@10} & \\textbf{Hit@20} & \\textbf{MRR} & \\textbf{Latency} \\\\\n")
        f.write("        \\midrule\n")
        for m in methods:
            s = summary[m]
            bold_start = "\\textbf{" if "Hybrid" in m else ""
            bold_end = "}" if "Hybrid" in m else ""
            f.write(f"        {m} & {s['Hit@1']:.1f}\\% & {s['Hit@5']:.1f}\\% & {s['Hit@10']:.1f}\\% & {s['Hit@20']:.1f}\\% & {bold_start}{s['MRR']:.3f}{bold_end} & {s['Avg_Latency_ms']:.1f}ms \\\\\n")
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("```\n")
        
    print(f"\nResults and LaTeX snippet saved to: {RESULTS_MD_PATH}")

if __name__ == "__main__":
    run_evaluation(test_llm_traps=True)

