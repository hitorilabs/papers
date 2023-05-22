import pandas as pd
import numpy as np
import pathlib
import os

# one-liner for copying data 
# for((i=1; i <= 100; i++)); do cp source_iris/iris.data "iris_dataset/iris_${i}.data"; done
source_path = os.getenv("SOURCE_PATH")
if source_path is None: raise Exception("Missing SOURCE_PATH variable")

pattern = os.getenv("PATTERN")
if pattern is None: raise Exception("Missing PATTERN variable")

source_path = pathlib.Path(".") / source_path
data_path = pathlib.Path("datasets") / source_path.stem
data_path.mkdir(exist_ok=True)

rows = 0
cols = 0
for i, file in enumerate(source_path.glob(pattern)):
    df = pd.read_csv(file, header=None)
    df_rows, df_cols = df.shape
    print(df.shape, i)
    rows += df_rows
    cols = df_cols

train_file = np.memmap(data_path / "train.memmap", dtype='float32', mode='w+', shape=(rows, cols -1))
target_file = np.memmap(data_path / "target.memmap", dtype='int64', mode='w+', shape=(rows))

current_rows = 0
for i, file in enumerate(source_path.glob(pattern)):
    df = pd.read_csv(
            file, 
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

    rows, cols = xs.shape
    left_pos = current_rows
    right_pos = current_rows + rows

    train_file[left_pos:right_pos,:cols] = xs[:rows,:cols]
    train_file.flush()

    target_file[left_pos:right_pos] = ys[:rows]
    target_file.flush()

    current_rows += rows
    print("loaded", i)