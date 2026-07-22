import os
import sys
import time
import math
import random
import argparse
import logging
import wandb
import numpy as np
from dataclasses import asdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import contextlib
from ..data.loader import DataLoader
from ..models.gpt2 import GPT2
from ..config.gpt2 import GPT2_124M_CFG
from .config import TrainConfig
from .evaluate import evaluate_validation_loss




parser = argparse.ArgumentParser(description="Train GPT2")
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--run-name", required=True)
parser.add_argument("--resume", action="store_true")


def set_seed(base_seed=1337, ddp_rank=0):
    # Offset the seed by rank for each process
    seed = base_seed + ddp_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_lr(step: int, train_cfg : TrainConfig):
    if step < train_cfg.warmup_steps: # Linear warmup
        return train_cfg.max_lr * (step + 1) / train_cfg.warmup_steps
    ratio = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps) # [0 - 1.0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return train_cfg.min_lr + coeff * (train_cfg.max_lr - train_cfg.min_lr)


def main():
    train_cfg = TrainConfig()
    model_cfg = GPT2_124M_CFG

    args = parser.parse_args()

    for key, value in vars(args).items():
        attr = key.replace("-", "_")
        if hasattr(train_cfg, attr):
            setattr(train_cfg, attr, value)

    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend='nccl')         # Initialize the distributed process group for DDP
        ddp_rank = int(os.environ['RANK'])              # Global ID across all nodes
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # ID on this specific machine
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total number of GPUs in the cluster

        # Bind this specific process to iIts designated GPU
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        # Setting the master process (only the master process print, log and save the checkpoints)
        master_process = ddp_rank == 0
    else:
        # Single GPU Fallback
        ddp_rank = 0 
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda"
        print(f"Warning!!! Not using DDP. Running on a single GPU")

    assert train_cfg.global_batch_size % (train_cfg.micro_batch_size * ddp_world_size) == 0, \
        "Global batch size must be cleanly divisible by (micro_batch_size * world_size)"
     
    # Replication
    set_seed(base_seed=1337, rank=ddp_rank)

    grad_accum_steps = train_cfg.global_batch_size // (train_cfg.batch_size * ddp_world_size)

    if master_process:
        print(f"Global batch size: {train_cfg.global_batch_size}")
        print(f"Batch size: {train_cfg.batch_size}")
        print(f"Gradient accumulation steps per GPU: {grad_accum_steps}")
    
    # Paths
    data_dir     = args.dataset_path
    out_dir      = args.our_dir
    log_path     = f"{out_dir}/train.log"
    ckpt_path    = f"{out_dir}/ckpt_latest.pt"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize the logger
    logger = logging.getLogger(__name__)

    if master_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[ 
                logging.FileHandler(log_path),  
                logging.StreamHandler()         
            ]
        )
    
    train_loader = DataLoader(
        data_path=data_dir,
        split="train",
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
        device=device,
        process_rank=ddp_rank,
        num_processes=ddp_world_size
    )

    val_loader = DataLoader(
        data_path=data_dir,
        split="val",
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
        device=device,
        process_rank=ddp_rank,
        num_processes=ddp_world_size
    )

    # Model Initialization
    model = GPT2(model_cfg).to(device)

    raw_model = model # Pure reference for clean checkpoints

    if master_process:
        params_count = sum(p.numel() for p in model.parameters())
        logger.info(f"{params_count/1e6:.2f}M Parameters")

    # Parameter separation for proper weight decay.
    decay_params = [
        p for p in model.parameters() 
        if p.requires_grad() and p.dim() >= 2
    ]
    nodecay_params = [
        p for p in model.parameters()
        if p.requires_grad() and p.dim() < 2
    ]

    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=train_cfg.max_lr, fused=True)
    
    # Mixed precision training context manager
    scaler = torch.amp.GradScaler()

    # Initializing va
    start_step = 0
    best_val_loss = float('inf')
    running_train_loss = 0.0

    # Resume logic
    if args.resume:
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint["step"] + 1
            train_loader.load_state_dict(checkpoint["train_loader"])
            scaler.load_state_dict(checkpoint["scaler"])
            best_val_loss = checkpoint["best_val_loss"]
            running_train_loss = checkpoint["running_train_loss"]
            wandb_id = checkpoint.get("wandb_id", None)
        else:
            if master_process:
                logger.info(f"Resume flag set, but no checkpoint found at {ckpt_path}")
            sys.exit()

    if master_process:
        wandb.init(
            project="GPT2",
            name=args.run_name,
            id=wandb_id if wandb_id else None,
            resume="allow" if wandb_id else None,
            config={
                **asdict(model_cfg),
                **asdict(train_cfg),
                "grad_accum_steps": grad_accum_steps,
            }
        )

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    model = torch.compile(model)

    # toks/sec
    t0 = time.perf_counter()

    for step in range(start_step, train_cfg.max_steps):

        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_accum = torch.tensor(0.0, device=device)

        # Gradient accumulation 
        for accum_step in range(grad_accum_steps):
            xb, yb = train_loader.next_batch() 
            is_last_step = (accum_step == grad_accum_steps - 1)
            
            # To prevent recompuation of gradients
            context = model.no_sync() if ddp and not is_last_step else contextlib.nullcontext()
            
            with context: 
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(xb, yb)
                    loss = loss / grad_accum_steps
            
                # Scaling the gradients as we are using fp16
                scaled_loss = scaler.scale(loss) # loss is scaled and
                scaled_loss.backward() # as a result the gradients (param.grad) are also scaled which prevents underflow
                loss_accum += loss.detach()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        loss_accum = loss_accum.item()

        if  master_process:
            if step == 0:
                running_train_loss = loss_accum
            else:
                running_train_loss = 0.99 * running_train_loss + 0.01 * loss_accum
            
            wandb.log({
                "train/loss": loss_accum,
                "train/lr": lr,
            }, step=step)
        

        # Unscale the gradients before clipping
        scaler.unscale_(optimizer)
        # Clip gradients and step the optimizer once per large batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Skips the optimization if gradients overflow or underflow
        scaler.step(optimizer) 
        # Reduces the scale automatically if gradients overflow
        # Increases the scale if multiple steps are succesfull to prevent better underflow
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  

        if step == start_step:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            if master_process:
                logger.info(
                    f"Step {step:5d} | "
                    f"Init Loss: {loss_accum:.4f} | "
                    f"LR: {lr:.4e} | "
                    f"(Compiling model, skipping perf metrics)"
                )

        elif step % train_cfg.stats_iter == 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1 # Reset the clock
            if master_process:

                tokens_processed = train_cfg.global_batch_size * model_cfg.block_size * train_cfg.stats_iter
                tokens_per_sec = tokens_processed / dt


                wandb.log({
                    "perf/toks_sec": tokens_per_sec,
                }, step=step)

                logger.info(
                    f"Step {step:5d} | "
                    f"Loss: {running_train_loss:.4f} | "
                    f"LR: {lr:.4e} | "
                    f"Tok/s: {tokens_per_sec:.0f}"

                )


        # Evaluation and Checkpointing
        if step >  0 and (step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps - 1):
            
            val_loss = evaluate_validation_loss(model, val_loader, train_cfg.eval_iters)

            if master_process:
                logger.info(f"EVAL | Step: {step:5d}  | Val Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss,}, step=step)
                
                is_best = val_loss < best_val_loss

                if is_best:
                    best_val_loss = val_loss

                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_loader": train_loader.state_dict(),
                    "step": step,
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(train_cfg),
                    "best_val_loss": best_val_loss,
                    "running_train_loss": running_train_loss,
                    "wandb_id": wandb.run.id if wandb is not None else None
                }

                # Always save the latest state for crash recovery
                torch.save(checkpoint, f"{out_dir}/ckpt_latest.pt")

                # Save if the model has a better val_loss
                if is_best:
                    torch.save(checkpoint, f"{out_dir}/ckpt_best.pt")
                    logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            
            if ddp:
                dist.barrier()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()    
                
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()