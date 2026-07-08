"""
Simple CLI for chatting with a trained GPT2 checkpoint.

Usage:
    python inference.py --model-path /path/to/ckpt_best.pt
"""

import argparse
import torch
from transformers import GPT2TokenizerFast

from .model import GPT2, GPTConfig


def load_model(model_path, device):
    # weights_only=False because our training checkpoints bundle more than
    # just tensors (optimizer state, step count, etc.) - see train.py
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Checkpoints from train.py store the dataclass as a plain dict under "config"
    config = GPTConfig(**checkpoint["config"])

    model = GPT2(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()  # disable dropout etc. (no-op here, but good habit)

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained GPT2 model")
    parser.add_argument("--model-path", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} onto {device}...")

    model, config = load_model(args.model_path, device)
    enc = GPT2TokenizerFast.from_pretrained("gpt2")  # same vocab/merges as tiktoken's "gpt2"
    eot_token_id = enc.eos_token_id  # <|endoftext|> - id 50256 for gpt2, marks end of generation

    print("Model loaded. Type a prompt and press enter. Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break

        # Encode prompt -> tensor of shape (1, T)
        tokens = enc.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        print("Model: ", end="", flush=True)

        # generate() yields one new token id at a time, so we can print as we go
        for next_token in model.generate(idx, args.max_new_tokens, args.temperature):
            token_id = next_token.item()
            if token_id == eot_token_id:
                break  # model signaled it's done; no need to burn through max_new_tokens
            print(enc.decode([token_id]), end="", flush=True)

        print("\n")


if __name__ == "__main__":
    main()
