from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int
    n_embd: int
    n_head: int
    n_layer: int


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head

        # Key, query, value projections for all heads, combined in a single linear layer
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)

        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.SCALE_INIT = True

        # Causal Mask (Not needed with flash attention)
        # self.register_buffer("tril", torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)), persistent=False)


    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        # Manual implementation of scaled dot-product attention with causal masking
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        # scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # type: ignore
        # attention = F.softmax(scores, dim=-1) # (B, nh, T, T)
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # y = attention @ v

        # Fused causal attention (SDPA / FlashAttention)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)

        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        out = out.transpose(1, 2).reshape(B, T, C)

        # Output projection (B, T, C) @ (C, C) -> (B, T, C)
        return self.proj(out)


class MLP(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )
        self.net[2].SCALE_INIT = True  

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Token + Positional embeddings
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.n_embd)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
         # Weight tying (Reduces the param count and force the model to learn a better representation)
        self.lm_head.weight = self.token_embedding.weight 

        self.register_buffer(
            "position_ids", torch.arange(cfg.max_seq_len), persistent=False
        )

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        # Residual projections (proj, MLP.net[2]) are scaled by 1 / sqrt(2 * n_layer)
        # because each block adds 2 residual branches (+attn, +mlp) to x, keeping residual variance ~1.
        if isinstance(module, nn.Linear):
            std = 0.02
        
            if getattr(module, "SCALE_INIT", False):
                std *= (2 * self.cfg.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.token_embedding(idx) + self.pos_embedding(self.position_ids[:T])
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B*T, logits.shape[-1]), targets.reshape(B*T))
        return logits, loss


    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Streams tokens one step at a time. Yields the new token (B, 1) each step."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next