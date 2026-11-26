import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from llm_forge.config import load_config
from llm_forge.data.loader import DataLoader
from llm_forge.trainer.config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train a language model")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoints")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--data-dir", default=None, help="Optional override for data.data_dir")
    parser.add_argument("--resume", action="store_true", help="Resume from ckpt_latest.pt")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to lr * min_lr_ratio."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    min_lr = cfg.lr * cfg.min_lr_ratio
    decay_steps = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (cfg.lr - min_lr)

@torch.inference_mode()
def evaluate(model, val_loader: DataLoader, eval_iters: int, device: str) -> float:
    model.eval()
    val_loader.reset()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = val_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def main():
    args = parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir

    model_cfg, data_cfg, train_cfg = cfg.model, cfg.data, cfg.train
    assert data_cfg.seq_len <= model_cfg.max_seq_len, (
        f"data.seq_len ({data_cfg.seq_len}) > model.max_seq_len ({model_cfg.max_seq_len})"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    set_seed(train_cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_latest = out_dir / "ckpt_latest.pt"
    ckpt_best = out_dir / "ckpt_best.pt"

    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum_steps * data_cfg.seq_len
    print(
        f"Arch: {cfg.arch} | Device: {device} | Seq len: {data_cfg.seq_len} | "
        f"Batch: {train_cfg.batch_size} x {train_cfg.grad_accum_steps} "
        f"({tokens_per_step:,} tok/step)"
    )

    train_loader = DataLoader(
        data_cfg, split="train", batch_size=train_cfg.batch_size, device=device, seed=train_cfg.seed
    )
    val_loader = DataLoader(
        data_cfg, split="val", batch_size=train_cfg.batch_size, device=device, seed=train_cfg.seed
    )

    raw_model = cfg.build_model().to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    start_step = 0
    best_val_loss = float("inf")
    wandb_id = None

    if args.resume:
        ckpt = torch.load(ckpt_latest, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["best_val_loss"]
        wandb_id = ckpt.get("wandb_id")
        print(f"Resumed from {ckpt_latest} at step {start_step}")


    wandb.init(
        project="llm-forge",
        name=args.run_name,
        id=wandb_id,
        resume="allow" if wandb_id else None,
        config=cfg.to_dict(),
        mode="disabled" if args.no_wandb else None,
    )

    model = torch.compile(raw_model)

    for step in range(start_step, train_cfg.max_steps):
        t0 = time.perf_counter()

        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_accum = torch.zeros((), device=device)

        for _ in range(train_cfg.grad_accum_steps):
            xb, yb = train_loader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(xb, yb)
                loss = loss / train_cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_cfg.log_interval == 0:
            loss_val = loss_accum.item()
            grad_norm_val = grad_norm.item()
            dt = time.perf_counter() - t0
            tok_per_sec = tokens_per_step / dt

            wandb.log(
                {
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_val,
                    "perf/tok_per_sec": tok_per_sec,
                },
                step=step,
            )
            print(
                f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e} | "
                f"norm {grad_norm_val:.2f} | {dt * 1000:.0f}ms | {tok_per_sec:,.0f} tok/s"
            )

        if step > start_step and (
            step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1
        ):
            val_loss = evaluate(model, val_loader, train_cfg.eval_iters, device)
            wandb.log({"val/loss": val_loss}, step=step)
            print(f"EVAL | step {step:5d} | val_loss {val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loader": train_loader.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "config": cfg.to_dict(),
                "wandb_id": wandb.run.id if (wandb.run and not args.no_wandb) else None,
            }
            torch.save(checkpoint, ckpt_latest)
            if is_best:
                torch.save(checkpoint, ckpt_best)
                print(f"New best checkpoint saved (val_loss {best_val_loss:.4f})")

    wandb.finish()

if __name__ == "__main__":
    main()
