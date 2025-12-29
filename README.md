# LLM Inference Optimizations

This repository contains implementations of key inference optimization techniques for Large Language Models (LLMs).

## Components

### 1. KV Cache Implementation (`kv_cache_implementation.py`)
A from-scratch implementation of Key-Value (KV) Caching for a GPT-style transformer.
- **Features:** 
  - Custom `unoptimized_gpt` vs `cached_gpt` comparison.
  - Implements the caching mechanism in the attention layer.
  - Demonstrates significant speedup in autoregressive generation.
- **Base Model:** GPT-2 XL.

### 2. Speculative Decoding (`speculative_decoding/`)
Implementation of Speculative Sampling to accelerate inference without degrading model quality.
- **Algorithm:** Based on [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (DeepMind).
- **Logic:** Uses a smaller "draft" model to propose tokens and a larger "target" model to verify them in parallel.
- **Key Files:**
  - `specdec.py`: Core logic for `sd_sample` (speculative sampling) and `ar_sample` (autoregressive sampling).
  - `benchmark.py`: Script to compare speed and match rate between standard and speculative decoding.

## Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run KV Cache Demo:**
   ```bash
   python kv_cache_implementation.py
   ```
   *Note: This script compares the generation time of standard vs cached implementation.*

3. **Run Speculative Decoding Benchmark:**
   ```bash
   cd speculative_decoding
   python benchmark.py
   ```

## References
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
