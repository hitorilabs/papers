import torch
from torch.utils.data import DataLoader, Dataset
import time

BATCH_SIZE = 32
NUM_WORKERS = 4
DATA_SHAPE = (3,224, 224)


for _ in range(10):
    # BATCH * CHANNELS * H * W
    data = torch.randn(BATCH_SIZE, *DATA_SHAPE, dtype=torch.float16)
    start = time.perf_counter()
    data.to("cuda:0")
    end = time.perf_counter()

    elapsed = end - start
    data_size = data.nelement() * data.element_size()
    throughput = data_size / elapsed / (1<<30)
    print(f"completed in {elapsed:.8f}s | {throughput=}GiB/s")
