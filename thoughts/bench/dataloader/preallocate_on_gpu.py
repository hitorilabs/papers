import torch
import time

BATCH_SIZE = 32
DATA_SHAPE = (3, 224, 224)

# Preallocate memory on the GPU
preallocated_data = torch.empty(BATCH_SIZE, *DATA_SHAPE, dtype=torch.float16, device="cuda:0")

for _ in range(10):
    data = torch.randn(BATCH_SIZE, *DATA_SHAPE, dtype=torch.float16)

    start = time.perf_counter()
    # Copy data to the preallocated space on the GPU
    preallocated_data.copy_(data)
    
    end = time.perf_counter()
    elapsed = end - start
    data_size = data.nelement() * data.element_size()
    throughput = data_size / elapsed / (1 << 30)
    print(f"completed in {elapsed:.8f}s | {throughput=}GiB/s")
