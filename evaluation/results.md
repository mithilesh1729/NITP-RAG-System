# Quantitative Evaluation Benchmark Results

**Test Suite Size:** 30 queries (25 factual benchmark queries + 5 negative guardrail trap queries)
**Indexed Documents:** 10 NIT Patna official circulars (76 chunks)
**Embedding Model:** `gemini-embedding-2` (3072-dimensional vectors)
**Evaluation Date:** 2026-10-16 18:33:43

## 1. Empirical Retrieval Performance

| Retrieval Engine | Hit@1 (%) | Hit@5 (%) | Hit@10 (%) | Hit@20 (%) | MRR | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Vector** | 68.0% | 80.0% | 84.0% | 84.0% | 0.716 | 583.6ms |
| **Sparse BM25** | 60.0% | 84.0% | 92.0% | 100.0% | 0.712 | 0.3ms |
| **Hybrid RRF (Proposed)** | 68.0% | 88.0% | 88.0% | 96.0% | 0.771 | 584.1ms |

**Zero-Hallucination Guardrail Rejection Rate:** `20.0%` (1/5 out-of-scope traps correctly refused).

## 2. LaTeX Table Snippet for Thesis & Slide 15

```latex
\begin{table}[h]
    \centering
    \caption{Empirical Retrieval Comparison on NIT Patna Institutional Dataset}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Retrieval Strategy} & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & \textbf{Hit@20} & \textbf{MRR} & \textbf{Latency} \\
        \midrule
        Dense Vector & 68.0\% & 80.0\% & 84.0\% & 84.0\% & 0.716 & 583.6ms \\
        Sparse BM25 & 60.0\% & 84.0\% & 92.0\% & 100.0\% & 0.712 & 0.3ms \\
        Hybrid RRF (Proposed) & 68.0\% & 88.0\% & 88.0\% & 96.0\% & \textbf{0.771} & 584.1ms \\
        \bottomrule
    \end{tabular}
\end{table}
```
