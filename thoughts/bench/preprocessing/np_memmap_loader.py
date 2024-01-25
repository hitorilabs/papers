from torch.utils.data import DataLoader, Dataset
import time
import torch
import numpy as np

from argparse import ArgumentParser

device = torch.device("cuda:0")

parser = ArgumentParser()
parser.add_argument("--workers", type=int, help="number of workers", default=4)
parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
parser.add_argument("--file", type=str, help="hdf5 file", default="data/train_images.dat")
parser.add_argument("--group", type=str, help="hdf5 group", default="train")
parser.add_argument("--cuda", action="store_true", help="move images onto device")
opt = parser.parse_args()

fp = np.memmap(opt.file, dtype="float32", mode="r", shape=(1281167, 3, 64, 64))
loader = DataLoader(fp, num_workers=opt.workers, batch_size=opt.batch_size, pin_memory=True)

print(f"bs={opt.batch_size}-{'cuda' if opt.cuda else 'cpu'}")
print(f"idx, time_s, throughput_s")
start = time.perf_counter()
for i, batch in enumerate(loader, 1):
    image = batch
    if opt.cuda:
        out = image.to(device) * 2 
        torch.cuda.synchronize()
    print(f"{i}, {(time.perf_counter() - start)}, {(i * opt.batch_size) / (time.perf_counter() - start)}")
torch.cuda.synchronize()
print(f"{i}, {(time.perf_counter() - start)}, {(i * opt.batch_size) / (time.perf_counter() - start)}")
