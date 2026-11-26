import argparse
from collections.abc import Generator
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


class ShardWriter:
    """Buffers uint16 token IDs and flushes fixed-size .bin shards to disk."""

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buf = np.empty(shard_size, dtype=np.uint16)
        self.pos = 0
        self.shard_idx = 0
        self.total_tokens = 0

    def write(self, tokens: np.ndarray) -> None:
        offset = 0
        while offset < len(tokens):
            take = min(self.shard_size - self.pos, len(tokens) - offset)
            self.buf[self.pos : self.pos + take] = tokens[offset : offset + take]
            self.pos += take
            offset += take
            self.total_tokens += take
            if self.pos == self.shard_size:
                self.flush()

    def flush(self) -> None:
        if self.pos == 0:
            return
        path = self.out_dir / f"shard_{self.shard_idx:04d}.bin"
        self.buf[: self.pos].tofile(path)
        print(f"Wrote {path} ({self.pos:,} tokens)")
        self.shard_idx += 1
        self.pos = 0


def stream_text_batches(
    dataset: str,
    subset: str | None,
    split: str,
    text_column: str,
    batch_size: int,
) -> Generator[list[str], None, None]:
    """Stream non-empty text batches from a Hugging Face dataset."""
    is_dir = subset is not None and "/" in subset
    stream = load_dataset(
        dataset,
        name=None if is_dir else subset,
        data_dir=subset if is_dir else None,
        split=split,
        streaming=True,
    )
    batch: list[str] = []
    for row in stream:
        text = row[text_column]
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_dataset(
    dataset: str,
    subset: str | None,
    split: str,
    text_column: str,
    tokenizer_path: str,
    out_dir: Path,
    shard_size: int,
    val_tokens: int,
    max_tokens: int | None,
    batch_size: int,
) -> None:
    """Stream, tokenize, and write train/val uint16 token shards."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, f"Tokenizer '{tokenizer_path}' has no eos_token_id."
    assert len(tokenizer) <= np.iinfo(np.uint16).max + 1, "Vocab exceeds uint16 range."

    val_writer = ShardWriter(out_dir / "val", min(shard_size, val_tokens)) if val_tokens > 0 else None
    train_writer = ShardWriter(out_dir / "train", shard_size)

    batches = stream_text_batches(
        dataset=dataset,
        subset=subset,
        split=split,
        text_column=text_column,
        batch_size=batch_size,
    )

    total_tokens = 0
    for texts in batches:
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        flat: list[int] = []
        for ids in encoded:
            flat.extend(ids)
            flat.append(eos_id)
        tokens = np.array(flat, dtype=np.uint16)

        if max_tokens is not None:
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

        if val_writer is not None and val_writer.total_tokens < val_tokens:
            need_val = val_tokens - val_writer.total_tokens
            val_writer.write(tokens[:need_val])
            tokens = tokens[need_val:]
            if val_writer.total_tokens == val_tokens:
                val_writer.flush()

        if len(tokens) > 0:
            train_writer.write(tokens)

        total_tokens = (val_writer.total_tokens if val_writer else 0) + train_writer.total_tokens
        if max_tokens is not None and total_tokens >= max_tokens:
            break

    batches.close()

    if val_writer is not None:
        val_writer.flush()
    train_writer.flush()

    val_total = val_writer.total_tokens if val_writer else 0
    print(f"Done. val={val_total:,} tokens, train={train_writer.total_tokens:,} tokens -> {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream, tokenize, and shard a Hugging Face dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset config name or data_dir.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Tokenizer path or HF name.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory (creates train/ and val/).")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per shard.")
    parser.add_argument("--val-tokens", type=int, default=10_000_000, help="Tokens reserved for val split.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional cap on total tokens.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Documents per tokenization batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        text_column=args.text_column,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        val_tokens=args.val_tokens,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
