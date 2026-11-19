import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import wandb

from ..config import build_model, load_config
from ..data.loader import DataLoader
from .config import TrainConfig


parser = argparse.ArgumentParser(description="Pre-train a language model from YAML config")
parser.add_argument("--config", required=True, help="Path to experiment YAML config")
parser.add_argument("--out-dir", required=True, help="Directory for logs and checkpoints")
parser.add_argument("--run-name", required=True, help="W&B run name")
parser.add_argument("--data-dir", default=None, help="Optional override for data.data_dir")
parser.add_argument("--resume", action="store_true", help="Resume from ckpt_latest.pt")


def set_seed(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_lr(step: int, train_cfg: TrainConfig):
    """Linear warmup + cosine decay learning rate schedule."""
    warmup_steps = train_cfg.warmup_steps or 0
    if warmup_steps > 0 and step < warmup_steps:
        return train_cfg.max_lr * (step + 1) / warmup_steps

    decay_steps = max(1, train_cfg.max_steps - warmup_steps)
    ratio = min(1.0, (step - warmup_steps) / decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return train_cfg.min_lr + coeff * (train_cfg.max_lr - train_cfg.min_lr)


@torch.inference_mode()
def evaluate_validation_loss(model, val_loader, eval_iters: int, device_type: str):
    model.eval()
    val_loader.reset()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = val_loader.next_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(xb, yb)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir

    model_cfg = cfg.model
    data_cfg = cfg.data
    train_cfg = cfg.train

    seq_len = data_cfg.seq_len or model_cfg.max_seq_len
    assert seq_len <= model_cfg.max_seq_len, (
        f"data.seq_len ({seq_len}) exceeds model.max_seq_len ({model_cfg.max_seq_len})"
    )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    ckpt_path = os.path.join(out_dir, "ckpt_latest.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    set_seed()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    assert train_cfg.global_batch_size % train_cfg.micro_batch_size == 0, (
        "global_batch_size must be cleanly divisible by micro_batch_size"
    )
    grad_accum_steps = train_cfg.global_batch_size // train_cfg.micro_batch_size

    logger.info(f"Arch: {cfg.arch} | Device: {device} | Seq len: {seq_len}")
    logger.info(
        f"Global batch size: {train_cfg.global_batch_size} | "
        f"Micro batch size: {train_cfg.micro_batch_size} | "
        f"Grad accum steps: {grad_accum_steps}"
    )

    train_loader = DataLoader(
        data_dir=data_cfg.data_dir,
        split="train",
        seq_len=seq_len,
        batch_size=train_cfg.micro_batch_size,
        device=device,
    )
    val_loader = DataLoader(
        data_dir=data_cfg.data_dir,
        split="val",
        seq_len=seq_len,
        batch_size=train_cfg.micro_batch_size,
        device=device,
    )

    model = build_model(cfg.arch, model_cfg).to(device)
    raw_model = model  # uncompiled reference for clean checkpointing

    params_count = sum(p.numel() for p in model.parameters())
    logger.info(f"{params_count / 1e6:.2f}M Parameters")

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=train_cfg.max_lr, fused=(device_type == "cuda")
    )

    start_step = 0
    best_val_loss = float("inf")
    running_train_loss = 0.0
    wandb_id = None

    if args.resume:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            train_loader.load_state_dict(checkpoint["train_loader"])
            start_step = checkpoint["step"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            running_train_loss = checkpoint["running_train_loss"]
            wandb_id = checkpoint.get("wandb_id")
            logger.info(f"Resumed from {ckpt_path} at step {start_step}")
        else:
            logger.info(f"Resume flag set, but no checkpoint found at {ckpt_path}")
            sys.exit(1)

    wandb.init(
        project="llm-forge",
        name=args.run_name,
        id=wandb_id,
        resume="allow" if wandb_id else None,
        config={
            **cfg.to_dict(),
            "grad_accum_steps": grad_accum_steps,
        },
    )

    model = torch.compile(model)

    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_accum = torch.tensor(0.0, device=device)

        for _ in range(grad_accum_steps):
            xb, yb = train_loader.next_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(xb, yb)
                loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()

        loss_accum = loss_accum.item()
        if step == 0:
            running_train_loss = loss_accum
            logger.info(f"Step {step:5d} | Init Loss: {loss_accum:.4f} | LR: {lr:.4e}")
        else:
            running_train_loss = 0.99 * running_train_loss + 0.01 * loss_accum

        wandb.log({"train/loss": loss_accum, "train/lr": lr}, step=step)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step > 0 and step % train_cfg.stats_iter == 0:
            if device_type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1

            tokens_processed = (
                train_cfg.global_batch_size * seq_len * train_cfg.stats_iter
            )
            tokens_per_sec = tokens_processed / dt

            wandb.log({"perf/toks_sec": tokens_per_sec}, step=step)
            logger.info(
                f"Step {step:5d} | "
                f"Loss: {running_train_loss:.4f} | "
                f"LR: {lr:.4e} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )

        if step > 0 and (
            step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1
        ):
            val_loss = evaluate_validation_loss(
                model, val_loader, train_cfg.eval_iters, device_type
            )
            logger.info(f"EVAL | Step: {step:5d} | Val Loss: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=step)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loader": train_loader.state_dict(),
                "step": step,
                "config": cfg.to_dict(),
                "best_val_loss": best_val_loss,
                "running_train_loss": running_train_loss,
                "wandb_id": wandb.run.id if wandb.run is not None else None,
            }

            torch.save(checkpoint, ckpt_path)
            if is_best:
                torch.save(checkpoint, os.path.join(out_dir, "ckpt_best.pt"))
                logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")

            if device_type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

    wandb.finish()


if __name__ == "__main__":
    main()
