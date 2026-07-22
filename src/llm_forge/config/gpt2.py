from ..models.gpt2 import GPTConfig




GPT2_124M_CFG = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
)

GPT2_355M_CFG = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=24,
    n_head=16,
    n_embd=1024,
)

GPT2_774M_CFG = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=36,
    n_head=20,
    n_embd=1280,
)

GPT2_1558M_CFG = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=48,
    n_head=25,
    n_embd=1600,
)