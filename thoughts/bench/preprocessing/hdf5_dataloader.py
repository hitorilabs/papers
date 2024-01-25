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
opt = parser.parse_args()

class H5Dataset(Dataset):
    def __init__(self, file_name, split="train"):
        with h5py.File(file_name, 'r') as f:
            self.length = len(f[split]["images"])
            self.img_hdf5 = h5py.File(file_name, 'r')
            self.dataset = self.img_hdf5[split] # if you want dataset.
    
    def __getitem__(self, index):
        record = self.h5_file[str(index)]
        return (
            record['data'].value,
            record['target'].value,
        )
        
    def __len__(self):
        return self.length

class LXRTDataLoader(Dataset):
    def __init__(self, file_name, split="train"):
        with h5py.File(file_name, 'r') as f:
            self.length = len(f[split]["images"])
            self.img_hdf5 = h5py.File(file_name, 'r')
            self.dataset = self.img_hdf5[split] # if you want dataset.

    def __getitem__(self, index: int):
        img0 = self.dataset["images"][index] # Do loading here
        img1 = self.dataset["labels"][index]
        return img0, img1
    
    def __len__(self):
        return self.length

train_ds = LXRTDataLoader(opt.file)
train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=32,
        num_workers=0
        )

for i in train_loader:
    print(i)
