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
    df.loc[:, df.columns != "class"].to_numpy(), 
    df["class"].cat.codes.to_numpy(dtype="int64"),
)

rows, *_ = xs.shape
fp = np.memmap(data_path / "train.memmap", dtype='float32', mode='w+', shape=xs.shape)
fp[:rows] = xs[:rows]
fp.flush()

rows, *_ = ys.shape
fp = np.memmap(data_path / "target.memmap", dtype="int64", mode='w+', shape=ys.shape)
fp[:rows] = ys[:rows]
fp.flush()