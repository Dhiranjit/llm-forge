# LLM-Forge

A modular PyTorch framework for pre-training decoder-only language models from scratch—covering custom tokenizer training, streaming binary dataset sharding, ratio-mixed multi-dataset loading, and config-driven `bfloat16` training.

## What's Built

- **Custom Byte-Level BPE Tokenizer (`data/train_tokenizer.py`):** Trained a 50,304-vocab Rust-backed Byte-Level BPE tokenizer streamed across ~1B characters of educational web text (`FineWeb-Edu`) and 7 programming/markup languages (`The Stack Smol`), featuring single-digit splitting for math and reserved chat/tool-calling/reasoning tokens (`<|im_start|>`, `<|tool_call|>`, `<|think|>`).
- **Streaming Data & Sharding Pipeline (`data/prepare.py`):** End-to-end streaming ingestion and tokenization pipeline that packs documents with `<|endoftext|>` boundaries into `uint16` memory-mappable binary shards (`train/` and `val/` splits) across multi-domain corpora (~1.55B tokens across `FineWeb-Edu`, `Cosmopedia-v2`, `The Stack` Python, and `FineMath`).
- **Resumable Multi-Dataset DataLoader (`data/loader.py`):** Zero-copy `np.memmap` token loader with per-dataset sequential shard readers, multinomial within-batch ratio mixing (e.g. 50% web, 25% synthetic textbooks, 15% code, 10% math), deterministic validation evaluation, pinned-memory async CUDA transfers, and exact checkpointable state (shard indices, byte offsets, and isolated RNG state).
- **GPT-2 Transformer Baseline (`models/gpt2.py`):** Clean from-scratch PyTorch implementation (124M-parameter config) using fused `F.scaled_dot_product_attention` (FlashAttention), pre-norm residual blocks, scaled residual weight initialization (`1 / sqrt(2N)`), tied input/output embeddings, and streaming autoregressive generation.
- **Training & Experiment Engine (`trainer/train.py`, `config.py`):** Typed YAML experiment configuration with `bfloat16` autocast, `torch.compile`, gradient accumulation, fused AdamW with parameter-group weight decay, linear warmup + cosine LR decay, gradient clipping, Weights & Biases logging, and fully resumable latest/best checkpointing.

**Next:** Implement the modern **Forge-360M** architecture (RoPE, RMSNorm, GQA, QK-Norm, SwiGLU) and benchmark it against the GPT-2 baseline.

---

## Usage

### Train the tokenizer

```bash
uv run python -m llm_forge.data.train_tokenizer \
  --source HuggingFaceFW/fineweb-edu:sample-10BT:text:750000000 \
  --source bigcode/the-stack-smol:data/python:content:70000000 \
  --source bigcode/the-stack-smol:data/javascript:content:40000000 \
  --source bigcode/the-stack-smol:data/typescript:content:40000000 \
  --source bigcode/the-stack-smol:data/c++:content:30000000 \
  --source bigcode/the-stack-smol:data/shell:content:25000000 \
  --source bigcode/the-stack-smol:data/sql:content:20000000 \
  --source bigcode/the-stack-smol:data/markdown:content:25000000 \
  --vocab-size 50304 \
  --out-dir tokenizer
```

### Prepare Dataset

```bash
uv run python -m llm_forge.data.prepare \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --tokenizer tokenizer \
  --out-dir data/fineweb_edu \
  --shard-size 100000000 \
  --val-tokens 10000000 \
  --max-tokens 1000000000
```

```bash
uv run python -m llm_forge.data.prepare \
 --dataset HuggingFaceTB/smollm-corpus \
 --subset cosmopedia-v2 \
 --text-column text \
 --tokenizer tokenizer \
 --out-dir data/cosmopedia_v2 \
 --shard-size 100000000 \
 --val-tokens 10000000 \
 --max-tokens 300000000
```

```bash
uv run python -m llm_forge.data.prepare \
  --dataset bigcode/the-stack-smol \
  --subset data/python \
  --text-column content \
  --tokenizer tokenizer \
  --out-dir data/stack_python \
  --shard-size 100000000 \
  --val-tokens 5000000 \
  --max-tokens 150000000
```

```bash
uv run python -m llm_forge.data.prepare \
  --dataset HuggingFaceTB/finemath \
  --subset finemath-4plus \
  --text-column text \
  --tokenizer tokenizer \
  --out-dir data/finemath \
  --shard-size 100000000 \
  --val-tokens 5000000 \
  --max-tokens 100000000
```

### Train Script

```bash
uv run python -m llm_forge.trainer.train \
  --config configs/test.yaml \
  --out-dir checkpoints/test \
  --no-wandb
```
