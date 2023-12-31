import h5py
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--index", type=int, help="retrieve index from hdf5")
parser.add_argument("--file", type=str, help="hdf5 file", default="imagenet.hdf5")
parser.add_argument("--group", type=str, help="hdf5 group", default="train")
opt = parser.parse_args()

filename = opt.file
with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("="*60)
    print(f"Group Keys: {f.keys()}")
    print(f"Selected: {opt.group}")
    print(f"""{f[opt.group]["images"].shape=}""")
    print(f"""{f[opt.group]["labels"].shape=}""")
    print("="*60 + "\n")
    image = f[opt.group]["images"][opt.index]
    print(image)
    print(f"{image.shape=}")
    label = f[opt.group]["labels"][opt.index]
    print(f"{label=}")
