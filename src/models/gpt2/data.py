import os
import glob
import torch
import numpy as np
import torch.distributed as dist



class DataLoader:
    """Sequential token data loader with sharding and DDP support."""
    def __init__(self, data_path, split, block_size, batch_size, device, process_rank=0, num_processes=1):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        # Distributed process
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Shard discovery (output: list of file paths as strings)
        self.shards = sorted(glob.glob(os.path.join(data_path, f"*{split}*.bin")))
        assert len(self.shards) > 0, f"No Shards for {split} split found in the {data_path}."

        if self.process_rank == 0: # master process
            print(f"Loaded {len(self.shards)} shards.")
        
        # State management 
        self.current_shard_idx = 0
        self.data = self._load_shard(self.current_shard_idx)
        
        # For cuda:0 -> current_position = 0 && cuda:1 -> current_postion = batch_size * block_size * 1
        # This makes sure that their batches never overlap.
        self.current_position = self.batch_size * self.block_size * self.process_rank


    def reset(self):
        self.current_shard_idx = 0
        self.data = self._load_shard(self.current_shard_idx)
        self.current_position = self.batch_size * self.block_size * self.process_rank


    def _load_shard(self, shard_idx):
        """Helper function to lazily load the data."""
        shard_path = self.shards[shard_idx]
        return np.memmap(shard_path, dtype=np.uint16, mode="r")

    def state_dict(self):
        """Returns the state of the dataloader."""
        return {
            "current_shard_idx": self.current_shard_idx,
            "current_position": self.current_position
        }

    def load_state_dict(self, state):
        """Restores the state of the dataloader."""
        self.current_shard_idx = state["current_shard_idx"]
        # Since the state is saved by Rank 0, we need to offset this for every rank.
        base_position = state["current_position"]
        self.current_position = base_position + (self.batch_size * self.block_size * self.process_rank)
        self.data = self._load_shard(self.current_shard_idx)

    def next_batch(self):
        """
        Sequentially load shards and sample batches sequentially.

        In DDP mode, each process reads from the same shards but starts at
        diffenrent offset to avoid overlapping samples.
        """
        # Tokens consumed by one GPU in a single batch
        B = self.batch_size
        T = self.block_size

        # Buffer from the current shard (for both x & y)
        buff = self.data[self.current_position : self.current_position + B * T + 1]

        x = torch.from_numpy(buff[:-1].astype(np.int64)).view(B, T).to(self.device)
        y = torch.from_numpy(buff[1: ].astype(np.int64)).view(B, T).to(self.device)

        # Advance the current_postion
        self.current_position += B * T * self.num_processes

        ### Shard transition
        # If the next batch results in an out-of bound error
        if self.current_position + (B * T * self.num_processes + 1) > len(self.data):
            # Advance to the next shard
            self.current_shard_idx += 1

            # If we hit the end of the shard then wrap around the beginning
            if self.current_shard_idx == len(self.shards):
                self.current_shard_idx = 0
            
            self.data = self._load_shard(self.current_shard_idx)
            # Reset the position for the new shard, according to the process rank
            self.current_position = B * T * self.process_rank

        return x, y


@torch.no_grad()
def evaluate_validation_loss(model,val_loader: DataLoader, total_eval_iters, process_rank, num_processes):
    model.eval()
    
    eval_iters_per_process = total_eval_iters // num_processes

    losses = torch.zeros(eval_iters_per_process, device=model.device)

    val_loader.reset()

    for k in range(eval_iters_per_process):
        x, y = val_loader.next_batch()
        _, loss = model(x, y)
        losses[k] = loss
    
    # Synchronize across GPUs
    mean_loss = losses.mean()
    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
    mean_loss /= num_processes

    model.train()
    return mean_loss.item()

        