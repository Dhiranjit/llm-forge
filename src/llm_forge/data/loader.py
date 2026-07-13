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
        # Distributed process context
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Shard discovery (self.shards: list of file paths as strings)
        self.shards = sorted(glob.glob(os.path.join(data_path, f"*{split}*.bin")))
        assert len(self.shards) > 0, f"No Shards for {split} split found in the {data_path}."

        if self.process_rank == 0: # master process
            print(f"Loaded {len(self.shards)} shards.")
        
        # State management 
        self.current_shard_idx = 0
        self.data = self._load_shard(self.current_shard_idx)
        
        # GLOBAL Position. Every rank starts exactly at 0.
        self.current_position = 0

        #### Removed the individual rank offset for a global current_postion.
        # # For cuda:0 -> current_position = 0 && cuda:1 -> current_postion = batch_size * block_size * 1
        # # This makes sure that their batches never overlap.
        # self.current_position = self.batch_size * self.block_size * self.process_rank
        ###

    def reset(self):
        self.current_shard_idx = 0
        self.data = self._load_shard(self.current_shard_idx)
        self.current_position = 0


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
        """Restores the state of the dataloader and safely re-applies DDP offsets."""
        self.current_shard_idx = state["current_shard_idx"]
        self.current_position = state["current_position"]
        self.data = self._load_shard(self.current_shard_idx)
        

    def next_batch(self):
        B, T = self.batch_size, self.block_size

        # Offset for this specific GPU.
        rank_offset = B * T * self.process_rank

        # Buffer from the current shard (for both x & y)
        buff = self.data[self.current_position + rank_offset : self.current_position + rank_offset +  B * T + 1]

        x_tensor = torch.from_numpy(buff[:-1].astype(np.int64)).view(B, T)
        y_tensor = torch.from_numpy(buff[1: ].astype(np.int64)).view(B, T)

        x = x_tensor.pin_memory().to(self.device, non_blocking=True)
        y = y_tensor.pin_memory().to(self.device, non_blocking=True)

        # Advance the global pointer by the total tokens consumed by all GPUS
        self.current_position += B * T * self.num_processes

        ### Shard transition
        # If the next global step will exceed the shard length
        # This guarantees all GPUs transition shards at the exact same moment
        if self.current_position + (B * T * self.num_processes + 1) > len(self.data):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.data = self._load_shard(self.current_shard_idx)
            self.current_position = 0

        return x, y