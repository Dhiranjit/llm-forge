import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

# REGEX pre-tokenization pattern with single-digit (\p{N}) splitting
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|bos|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_response|>",
    "<|/tool_response|>",
    "<|think|>",
    "<|/think|>",
]

@dataclass(frozen=True)
class DatasetSource:
    dataset: str
    subset: str | None
    text_column: str
    max_chars: int


def parse_source(spec: str) -> DatasetSource:
    """Parse a 'dataset:subset:text_column:max_chars' source string."""
    dataset, subset, text_column, max_chars = spec.split(":")
    return DatasetSource(
        dataset=dataset,
        subset=subset or None,
        text_column=text_column,
        max_chars=int(max_chars),
    )


def build_tokenizer() -> Tokenizer:
    """Create an untrained Byte-Level BPE tokenizer."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(SPLIT_PATTERN),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def stream_batches(
    sources: list[DatasetSource],
    split: str,
    batch_size: int,
) -> Iterator[list[str]]:
    """Stream text batches sequentially from one or more Hugging Face datasets."""
    batch: list[str] = []

    for src in sources:
        print(f"Streaming {src.dataset} (subset={src.subset}, max_chars={src.max_chars:,})...")
        is_dir = src.subset is not None and "/" in src.subset
        stream = load_dataset(
            src.dataset,
            name=None if is_dir else src.subset,
            data_dir=src.subset if is_dir else None,
            split=split,
            streaming=True,
        )
        total_chars = 0

        for row in stream:
            text = row[src.text_column]
            if not text:
                continue
            batch.append(text)
            total_chars += len(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if total_chars >= src.max_chars:
                break

    if batch:
        yield batch


def train_tokenizer(
    batches: Iterator[list[str]],
    vocab_size: int,
    min_frequency: int,
    out_dir: Path,
) -> Tokenizer:
    """Train a Byte-Level BPE tokenizer and save it in Hugging Face format."""
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(batches, trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    hf_tokenizer.save_pretrained(out_dir)
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom Byte-Level BPE tokenizer.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Dataset source as 'dataset:subset:text_column:max_chars' (repeatable).",
    )

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("tokenizer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [parse_source(s) for s in args.source]
    batches = stream_batches(sources=sources, split=args.split, batch_size=args.batch_size)
    tokenizer = train_tokenizer(
        batches=batches,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        out_dir=args.out_dir,
    )
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {args.out_dir}")


if __name__ == "__main__":
    main()
