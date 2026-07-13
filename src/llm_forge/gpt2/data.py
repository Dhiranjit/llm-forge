import os
import glob
import torch
import numpy as np
import torch.distributed as dist




        

@torch.no_grad()
def evaluate_validation_loss(model, val_loader, eval_iters):
    model.eval()

    losses = torch.zeros(eval_iters, device=val_loader.device)
    val_loader.reset()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for k in range(eval_iters):
            x, y = val_loader.next_batch()
            _, loss = model(x, y)
            losses[k] = loss

    mean_loss = losses.mean()

    if dist.is_initialized():
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= dist.get_world_size()

    model.train()
    return mean_loss.item()

        