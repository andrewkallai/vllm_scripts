import os
import time
import torch
import torch.distributed as dist
import ray

# 1. Initialize Ray cluster
ray.init()

# Define the number of GPUs you want to test
NUM_GPUS = 2  

@ray.remote(num_gpus=1)
class NCCLWorker:
    def __init__(self, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

    def run_allreduce(self):
        # Set up standard PyTorch distributed environment variables
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        
        # Enable full NCCL logging for debugging
        os.environ["NCCL_DEBUG"] = "INFO"
        
        # Initialize the process group with NCCL
        dist.init_process_group(backend="nccl", init_method="env://")
        
        # Move a test tensor to the designated GPU
        device = torch.device(f"cuda:0") # Ray remaps visible devices per actor
        tensor = torch.ones(100_000_000, device=device) * (self.rank + 1)
        
        print(f"[Rank {self.rank}] Before All-Reduce sum: {tensor[0].item()}")
        
        # Execute NCCL All-Reduce
        start_time = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        duration = time.time() - start_time
        
        print(f"[Rank {self.rank}] After All-Reduce sum: {tensor[0].item()} (Took {duration:.4f}s)")
        
        # Clean up
        dist.destroy_process_group()
        return f"Rank {self.rank} completed successfully."

# 2. Get the head node IP address dynamically for master orchestration
head_node_ip = ray.util.get_node_ip_address()
print(f"Head node IP address: {head_node_ip}",flush=True)
master_port = 29500

# 3. Spawn workers across the cluster
workers = [
    NCCLWorker.remote(rank=i, world_size=NUM_GPUS, master_addr=head_node_ip, master_port=master_port)
    for i in range(NUM_GPUS)
]

# 4. Trigger the test concurrently
print(f"Starting NCCL test on {NUM_GPUS} GPUs...")
results = ray.get([w.run_allreduce.remote() for w in workers])
print("Test finished results:")
print(results)
