import h5py
from torch.utils.data import DataLoader, Dataset
import time
import torch

from argparse import ArgumentParser

device = torch.device("cuda:0")

parser = ArgumentParser()
parser.add_argument("--workers", type=int, help="number of workers", default=4)
parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
parser.add_argument("--file", type=str, help="hdf5 file", default="preprocessed_imagenet.hdf5")
parser.add_argument("--group", type=str, help="hdf5 group", default="train")
parser.add_argument("--cuda", action="store_true", help="move images onto device")
opt = parser.parse_args()

with h5py.File(opt.file, "r") as f:
    start = time.perf_counter()
    print(f"bs={opt.batch_size}-{'cuda' if opt.cuda else 'cpu'}")
    print(f"idx, time_s, throughput_s")
    for i in range(0, f[opt.group]['images'].shape[0], opt.batch_size):
        image = torch.from_numpy(f[opt.group]["images"][i:i + opt.batch_size])
        if opt.cuda:
            image.to(device)
        label = f[opt.group]["labels"][i:i + opt.batch_size]

        print(f"{i}, {(time.perf_counter() - start)}, {(i + opt.batch_size) / (time.perf_counter() - start)}")
