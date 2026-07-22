from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass
class GPTConfig:
    vocab_size : int 
    block_size : int 
    n_embd     : int 
    n_head     : int
    n_layer    : int



class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.n_embd   = cfg.n_embd
        self.n_head    = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head

        # Key, query, value projections for all heads, combined in a single linear layer
        # (C, hs) for each of k, q, v in a single head -> (C, hs * nh) -> (C, C) for each k, q, v
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)

        # Output projection (C, C)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj.SCALE_INIT = True # Scale residual projection init by 1/sqrt(2*n_layer)

        # Causal Mask (Not needed with flash attention)
        # self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)), persistent=False)
    
    def forward(self, x):
        B, T, C = x.shape

        # (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x) 

        # Shapes: # (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1) 

        # Shapes: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        ### Manual implementation of scaled dot-product attention with causal masking
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        # scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # type: ignore
        # attention = F.softmax(scores, dim=-1) # (B, nh, T, T)

        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs) 
        # y = attention @ v
        
        # Switching to flash attention for fused kernel implementation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)

        
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection (B, T, C) @ (C, C) -> (B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )
        self.net[2].SCALE_INIT = True # Scale residual projection init by 1/sqrt(2*n_layer)
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # Layer Normalization
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

        # Core sub-layers
        self.attn  = CausalSelfAttention(cfg)
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
        self.token_embedding  = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding    = nn.Embedding(cfg.block_size, cfg.n_embd)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.n_layer)]
        )
        
        # Final LayerNorm and Language Model Head
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight # Weight tying (Reduces the param count and force the model to learn a better representation)

        self.register_buffer("position_ids", torch.arange(cfg.block_size), persistent=False)

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
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

        # (B, T) -> (B, T, C)
        tok_emb = self.token_embedding(idx)
        # (T,) -> (T, C)
        pos_emb = self.pos_embedding(self.position_ids[:T])

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            return logits, None 
        
        # Flatten Time + Batch for cross entropy
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T))
        return logits, loss
    
    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Streams tokens one step at a time. Yields the new token (B, 1) each step."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # Take only the last time step (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next