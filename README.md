# BoA-Based Post-Training Quantization of OPT-125M

The project explores a practical implementation of **BoA (Blockwise Hessian Approximation)** for post-training quantization of large language models, specifically **OPT-125M**, and compares it with the GPTQ baseline. The implementation is adapted to run efficiently on **Kaggle GPU (Tesla P100)** under memory constraints.

## Repository Contents

### 1. `BoA-Based Post-Training Quantization of OPT-125M.pdf`
This paper provides:

- Background on GPTQ and BoA
- Mathematical formulation of BoA Hessian
- Deviations from the official BoA algorithm  
  (including simplified Hessian, min-max quantization grids, partial row updates, etc.)
- Optimizations enabling quantization on Kaggle P100 GPU
- Experimental evaluations:  
  - PPL (WikiText-2)  
  - ARC-Easy / ARC-Challenge  
  - SQuAD-F1  
  - CNN/DailyMail BLEU & ROUGE-L  
- Comparison between FP16, INT4, and INT2 quantized OPT models  
- Discussion on performance gaps relative to the official BoA implementation  
- Guidelines for future improvements

---

### 2. `boa-implementation.ipynb`
A full Jupyter notebook implementing:

- OPT-125M loading in mixed CPU–GPU mode  
- Calibration dataset extraction (WikiText-2)
- Column Hessian estimation (GPTQ style)
- Partial BoA row-wise Hessian estimation for Q/K projections
- Row-coupled weight updates across attention heads
- Per-group quantization (MinMax uniform grid)
- Layer-by-layer quantization pipeline designed for low-memory environments
- Evaluation suite for:
  - Perplexity
  - ARC-Easy / ARC-Challenge
  - SQuAD-F1 (QA)
  - CNN/DailyMail summarization metrics

**Note:**  
This is a *relaxed* and *simplified* BoA implementation intended for research and educational purposes, not a reproduction of the official BoA codebase.
