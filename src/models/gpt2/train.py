import os
import time
import math
import random
import argparse
import logging
import wandb
import numpy as np
import dataclasses
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import DataLoader, evaluate_validation_loss
from .model import GPT2, GPTConfig

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True







### Model Config
block_size = 1024
n_embd     = 768
n_head     = 12
n_layer    = 12

### TRAINING CONFIG
vocab_size_padded = 50304           # GPT2Tokenizer vocab_size + padding
max_steps         = 35_000
eval_interval     = 1000
eval_iters        = 100
global_batch_size = 256
batch_size        = 16
max_lr            = 6e-4
min_lr            = 6e-5
warmup_steps      = max_steps // 20 # 5% of max_steps




parser = argparse.ArgumentParser(description="Train GPT2")
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--resume", action="store_true")


def set_seed(base_seed=1337, rank=0):
    # Offset the seed for each process
    seed = base_seed + rank

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    random.seed(seed)
    np.random.seed(seed)


def get_lr(step):
    if step < warmup_steps: # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / (max_steps - warmup_steps) # [0 - 1.0]
    # 1/2 * (1 + cos(pi*ratio)) -> [1, 0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    args = parser.parse_args()

    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])              # Global ID across all nodes
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # ID on this specific machine
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total number of GPUs in the cluster

        # Bind this specific process to its designated GPU
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        # Setting the master process 
        # Only the master process print, log and save the checkpoints
        master_process = ddp_rank == 0
    else:
        # Fallback
        ddp_rank = 0 
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
     

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Warning!!! Not using DDP. Running on {device}")

    assert global_batch_size % (batch_size * ddp_world_size) == 0, \
        "Global batch size must be cleanly divisible by (batch_size * world_size)"
    
    # Setting the seed for replication
    set_seed(rank=ddp_rank)

    # Dynamically calculate the the grad_accum_steps
    grad_accum_steps = global_batch_size // (batch_size * ddp_world_size)

    if master_process:
        print(f"Global batch size: {global_batch_size}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps per GPU: {grad_accum_steps}")
    
    # Paths
    data_dir     = args.dataset_path
    out_dir      = f"/kaggle/working/models/GPT2"
    log_path       = f"{out_dir}/train.log"
    os.makedirs(out_dir, exist_ok=True)


    
    # Find the nearest multiple of 64

    config = GPTConfig(
        vocab_size=vocab_size_padded, 
        block_size=block_size,
        n_embed=n_embd,
        n_head=n_head,
        n_layer=n_layer
    )


    if master_process:
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path),  
                logging.StreamHandler()         
            ]
        )

        # Initialize Weights & Biases
        wandb.init(
            project="gpt2",
            name=f"run-fineWebEdu-{max_steps}",
            config={
                "batch_size": batch_size,
                "block_size": config.block_size,
                "max_lr": max_lr,
                "vocab_size": config.vocab_size,
                "grad_accum_steps": grad_accum_steps
            }
        )
        
    logger = logging.getLogger(__name__)
    
    
    train_loader = DataLoader(
        data_path=data_dir,
        split="train",
        block_size=config.block_size,
        batch_size=batch_size,
        device=device,
        process_rank=ddp_rank,
        num_processes=ddp_world_size
    )

    val_loader = DataLoader(
        data_path=data_dir,
        split="val",
        block_size=config.block_size,
        batch_size=batch_size,
        device=device,
        process_rank=ddp_rank,
        num_processes=ddp_world_size
    )

    model = GPT2(config).to(device)

    if master_process:
        params_count = sum(p.numel() for p in model.parameters())
        print(f"{params_count/1e6:.2f}M Parameters")


    raw_model = model

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Separate parameters for proper weight decay application
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, fused=True)
    
    # Mixed precision training context
    scaler = torch.amp.GradScaler()

    # Training Loop
    start_step = 0
    best_val_loss = float('inf')
    running_train_loss = 0.0

    # Resume logic
    if args.resume:
        ckpt_path = f"{out_dir}/ckpt_latest.pt"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            if master_process:
                logger.info(f"Resuming training from {ckpt_path}")
            
            state_dict = checkpoint["model"]
            # Handle checkpoints saved from torch.compile()
            if next(iter(state_dict)).startswith("_orig_mod."):
                state_dict = {
                    k.replace("_orig_mod.", "", 1): v
                    for k, v in state_dict.items()
                }
            raw_model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])

            train_loader.load_state_dict(checkpoint["train_loader"])

            start_step = checkpoint["step"] + 1
            best_val_loss = checkpoint.get("val_loss", float('inf'))

            

        else:
            if master_process:
                logger.info(f"Resume flag set, but no checkpoint found at {ckpt_path}. Starting fresh training...")
    
    if device.startswith("cuda"):
        model = torch.compile(model)


    # toks/sec
    t0 = time.time()

    for step in range(start_step, max_steps):

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_accum = torch.tensor(0.0, device=device)

        # Gradient accumulation 
        for accum_step in range(grad_accum_steps):
            xb, yb = train_loader.next_batch() 
            is_last_step = (accum_step == grad_accum_steps - 1)
            
            if ddp:
                # Only sync gradients across GPUs on the last accum_step
                model.require_backward_grad_sync = is_last_step
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(xb, yb)
                loss = loss/grad_accum_steps
            
            # Scaling the gradients as we are using fp16
            scaled_loss = scaler.scale(loss) # loss is scaled and
            scaled_loss.backward() # as a result the gradients (param.grad) are also scaled which prevents underflow
            loss_accum += loss.detach()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        loss_accum = loss_accum.item()

        if  master_process:
            if step == start_step:
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
            t0 = time.time()
            
            if master_process:
                logger.info(
                    f"Step {step:5d} | "
                    f"Init Loss: {loss_accum:.4f} | "
                    f"LR: {lr:.4e} | "
                    f"(Compiling model, skipping perf metrics)"
                )

        elif step % 100 == 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1 # Reset the clock
            if master_process:

                tokens_processed = global_batch_size * config.block_size * 100
                tokens_per_sec = tokens_processed / dt

                peak_flops_promised = 65e12 * ddp_world_size

                flops_per_token = 6 * params_count
                observed_flops_per_sec = flops_per_token * tokens_per_sec
                mfu_percentage = (observed_flops_per_sec / peak_flops_promised) * 100

                wandb.log({
                    "perf/toks_sec": tokens_per_sec,
                    "perf/mfu": mfu_percentage
                }, step=step)

                logger.info(
                    f"Step {step:5d} | "
                    f"Loss: {running_train_loss:.4f} | "
                    f"LR: {lr:.4e} | "
                    f"Tok/s: {tokens_per_sec:.0f} | "
                    f"MFU: {mfu_percentage:.1f}%"
                )


        # Evaluation and Checkpointing
        if step >  0 and (step % eval_interval == 0 or step == max_steps - 1):
            
            val_loss = evaluate_validation_loss(
                model, 
                val_loader,
                eval_iters
            )

            if master_process:

                logger.info(f"EVAL | Step: {step:5d}  | Val Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss,}, step=step)
                
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_loader": train_loader.state_dict(),
                    "step": step,
                    "config": dataclasses.asdict(config),
                    "val_loss": val_loss
                }

                # Always save the latest state for crash recovery
                torch.save(checkpoint, f"{out_dir}/ckpt_latest.pt")

                # Save if the model has a better val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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