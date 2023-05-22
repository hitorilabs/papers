import pandas as pd
import numpy as np
import pathlib
import os

source_file = os.getenv("SOURCE_FILE")
if source_file is None: raise Exception("Missing SOURCE_FILE variable")
source_path = pathlib.Path(".") / source_file
data_path = pathlib.Path("datasets") / source_path.stem
data_path.mkdir(exist_ok=True)

df = pd.read_csv(
    source_path, 
    header=None,
    names=[
        "sepal_length", 
        "sepal_width", 
        "petal_length", 
        "petal_width", 
        "class"
    ],
    dtype={
        "sepal_length": np.float32, 
        "sepal_width": np.float32, 
        "petal_length": np.float32, 
        "petal_width": np.float32, 
        "class": "category",
    })

xs, ys = (
    np.repeat(df.loc[:, df.columns != "class"].to_numpy(),   1000000), 
    np.repeat(df["class"].cat.codes.to_numpy(dtype="int64"), 1000000),
)

print(xs.shape)
print(ys.shape)

np.save(data_path / "train_save.npy", xs)
np.save(data_path / "target_save.npy", ys)

rows, *_ = xs.shape
fp = np.memmap(data_path / "train_memmap.npy", dtype='float32', mode='w+', shape=xs.shape)
fp[:rows] = xs[:rows]
print(fp.shape)
fp.flush()

rows, *_ = ys.shape
fp = np.memmap(data_path / "target_memmap.npy", dtype="int64", mode='w+', shape=ys.shape)
print(fp.shape)
fp[:rows] = ys[:rows]
fp.flush()