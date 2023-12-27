import torch
import time

BATCH_SIZE = 32
DATA_SHAPE = (3, 224, 224)

# Preallocate memory on the GPU
preallocated_data = torch.empty(BATCH_SIZE, *DATA_SHAPE, device="cuda:0")

for _ in range(10):
    start = time.perf_counter()
    
    # Load tensor to CPU memory
    gpu_data = torch.load("batch.pt", map_location=torch.device("cuda:0"))

    end = time.perf_counter()

    elapsed = end - start
    data_size = gpu_data.nelement() * gpu_data.element_size()
    throughput = data_size / elapsed / (1 << 30)
    print(f"completed in {elapsed:.8f}s | {throughput=}GiB/s")

