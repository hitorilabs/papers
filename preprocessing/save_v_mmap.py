import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

data_path = pathlib.Path("datasets")
data_path.mkdir(exist_ok=True)

df = pd.read_csv(
    "1958-perceptron/iris.data", 
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
    },
    engine="pyarrow"
    )

xs, ys = (
    df.loc[:, df.columns != "class"].to_numpy(), 
    df["class"].cat.codes.to_numpy()
)


np.save(data_path / "train_save.npy", xs)
np.save(data_path / "target_save.npy", ys)

rows, *_ = xs.shape
fp = np.memmap(data_path / "train_memmap.npy", dtype='float32', mode='w+', shape=xs.shape)
fp[:rows] = xs[:rows]
fp.flush()

rows, *_ = ys.shape
fp = np.memmap(data_path / "target_memmap.npy", dtype="int64", mode='w+', shape=ys.shape)
fp[:rows] = ys[:rows]
fp.flush()