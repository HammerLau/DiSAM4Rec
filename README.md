# DiSAM4Rec: Distilled and Sparsity Adaptive Mamba for Sequential Recommendation

This is the official implementation of the paper: **"DiSAM4Rec: Distilled and Sparsity Adaptive Mamba for Sequential Recommendation"**.

## 🚀 Overview
DiSAM4Rec is a novel sequential recommendation framework designed to address user preference diversity and representation degradation in deep architectures. Our model integrates two core components:
1. **Multi-head Sparsity Adaptive Dual-branch (MSADB) Mamba**: Captures diverse behavioral patterns by dynamically adjusting sparsity.
2. **Layer-wise Semantic Alignment Self-Distillation (LSA-SD)**: Mitigates semantic inconsistency and performance decay as model depth increases.

## 📊 Experimental Results
We conducted 7 comprehensive experiments (including ablation studies, sensitivity analysis, and efficiency comparisons) across multiple public datasets (e.g., Beauty, Fashion, Yelp, and ML-1M) to verify the superiority of DiSAM4Rec.

| Datasets | Eval Metrics | GRU4Rec | BERT4Rec | SASRec | EchoMamba4Rec | Mamba4Rec | SIGMA | Ours | Improv. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | HIT@10 | 0.2990 | 0.2745 | 0.2990 | 0.3215 | 0.3098 | 0.3283 | **0.3344** | **1.86%** |
|  **ML-1M**  | NDCG@10 | 0.1716 | 0.1503 | 0.1693 | 0.1529 | 0.1776 | 0.1911 | **0.1973** | **3.24%** |
| | MRR@10 | 0.1329 | 0.1127 | 0.1298 | 0.1458 | 0.1374 | 0.1491 | **0.1559** | **4.56%** |
|  | HIT@10 | 0.0779 | 0.0545 | 0.1084 | 0.1103 | 0.1107 | 0.0918 | **0.1169** | **5.60%** |
|  **Beauty** | NDCG@10 | 0.0458 | 0.0274 | 0.0569 | 0.0590 | 0.0587 | 0.0541 | **0.0658** | **11.53%** |
| | MRR@10 | 0.0360 | 0.0193 | 0.0411 | 0.0443 | 0.0435 | 0.0427 | **0.0500** | **12.87%** |
| | HIT@10 | 0.0499 | 0.0489 | 0.0514 | 0.0680 | 0.0571 | 0.0629 | **0.0724** | **6.47%** |
|  **Yelp**   | NDCG@10 | 0.0267 | 0.0317 | 0.0387 | 0.0390 | 0.0311 | 0.0412 | **0.0491** | **19.17%** |
| | MRR@10 | 0.0197 | 0.0243 | 0.0349 | 0.0303 | 0.0233 | 0.0346 | **0.0420** | **20.34%** |
| | HIT@10 | 0.1288 | 0.0549 | 0.1776 | 0.2083 | 0.2049 | 0.2043 | **0.2136** | **2.54%** |
| **Fashion** | NDCG@10 | 0.0872 | 0.0238 | 0.1344 | 0.1523 | 0.1495 | 0.1534 | **0.1580** | **3.00%** |
| | MRR@10 | 0.0743 | 0.0146 | 0.1208 | 0.1345 | 0.1323 | 0.1374 | **0.1403** | **2.11%** |
  
@article{Liu2026DiSAM4Rec,
  title={DiSAM4Rec: Distilled and Sparsity Adaptive Mamba for Sequential Recommendation},
  author={Ruiming Liu, et al.},
  journal={Journal of Intelligent Information Systems},
  year={2026}
}
