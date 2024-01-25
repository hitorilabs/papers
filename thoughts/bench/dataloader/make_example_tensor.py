import torch

BATCH_SIZE = 32
DATA_SHAPE = (3, 224, 224)

preallocated_data = torch.randn(BATCH_SIZE, *DATA_SHAPE, dtype=torch.float16)
torch.save(preallocated_data, "batch.pt")
