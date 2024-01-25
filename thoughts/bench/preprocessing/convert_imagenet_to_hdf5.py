from datasets import load_dataset
from pathlib import Path
from argparse import ArgumentParser
import time

import torch
import numpy as np
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

dataset = load_dataset("imagenet-1k", cache_dir=DATA_PATH.as_posix(), trust_remote_code=False)

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

itos = dataset["train"].features["label"].int2str
stoi = dataset["train"].features["label"].str2int



NUM_CHANNELS = 3
TRANSFORMED_SHAPE = (NUM_CHANNELS, opt.imageSize, opt.imageSize)

import h5py
from tqdm import tqdm

with h5py.File('bench_imagenet.hdf5', 'a') as h5f:
    for split in dataset:
        # Count number of samples in the split
        num_samples = len(dataset[split])

        # Create a group for each split in the H5PY file
        group = h5f[split] if split in h5f else h5f.create_group(split)

        # Assuming a fixed image size, adjust as per actual size
        image_shape = TRANSFORMED_SHAPE  # Example shape, modify as needed
        dtype = np.float32  # Modify as per actual data type

        # Preallocate datasets for images and labels
        MAX_IMAGE_SHAPE = (num_samples, *TRANSFORMED_SHAPE)
        images_ds = group['images'] if 'images' in group else group.create_dataset('images', shape=MAX_IMAGE_SHAPE, maxshape=MAX_IMAGE_SHAPE, dtype=dtype)

        MAX_LABEL_SHAPE = (num_samples, )
        labels_ds = group['labels'] if 'labels' in group else group.create_dataset('labels', shape=MAX_LABEL_SHAPE, maxshape=MAX_LABEL_SHAPE, dtype=np.int64, compression='gzip')

        dataloader = DataLoader(dataset[split], batch_size=opt.batchSize, num_workers=opt.workers)
        start = time.perf_counter()
        if opt.csv:
            print(f"batch_start, batch_end, time_s, throughput_s, process_s, write_s")
        # Fill the datasets
        for i, batch in enumerate(dataloader):
            batch_start = i * opt.batchSize
            batch_end = min(batch_start + opt.batchSize, num_samples)
            process_s = f"{(time.perf_counter() - start)}"

            images_ds[batch_start:batch_end] = batch["image"].numpy()
            labels_ds[batch_start:batch_end] = batch["label"].numpy()
            write_s = f"{(time.perf_counter() - start)}"

            if opt.csv:
                print(f"{batch_start}, {batch_end}, {(time.perf_counter() - start)}, {batch_end / (time.perf_counter() - start)}, {process_s}, {write_s}")
            else:
                print(f"filling batch between {batch_start=} {batch_end=}")


