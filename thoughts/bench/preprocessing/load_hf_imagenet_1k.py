from datasets import load_dataset
from pathlib import Path
from argparse import ArgumentParser
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

device = torch.device("cuda:0")

parser = ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--csv',action='store_true',help='print format')

opt = parser.parse_args()

DATA_PATH = Path.home() / "datasets" / "huggingface"

dataset = load_dataset("imagenet-1k", cache_dir=DATA_PATH.as_posix())

tv_transforms = T.Compose([
    T.Lambda(lambda x: x.convert("RGB")),
    T.Resize(opt.imageSize),
    T.CenterCrop(opt.imageSize),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def transforms(batch):
    batch["image"] = tv_transforms(batch["image"])
    return batch
dataset.set_transform(transforms)

dataloader = DataLoader(dataset["train"], batch_size=opt.batchSize, num_workers=opt.workers)
# print(dataset)

if opt.csv:
    print("idx,images,img/s")

start = time.perf_counter()
for idx, batch in enumerate(dataloader):
    # batch["image"].to(device)
    # if idx > 0 and idx % 10 == 0:
    #     if opt.csv:
    #         print(f"{idx},{idx * opt.batchSize},{(idx * opt.batchSize) / (time.perf_counter() - start)}")
    #     else:
    #         print(f"idx: {idx} | images: {idx * opt.batchSize} | throughput {(idx * opt.batchSize) / (time.perf_counter() - start)} img/s")
    if opt.csv:
        print(f"{idx},{idx * opt.batchSize},{(idx * opt.batchSize) / (time.perf_counter() - start)}")
    else:
        print(f"idx: {idx} | images: {idx * opt.batchSize} | throughput {(idx * opt.batchSize) / (time.perf_counter() - start)} img/s")



