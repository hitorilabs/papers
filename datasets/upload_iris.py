from datasets import load_dataset, ClassLabel, Value, Features

dataset = load_dataset(
    "scikit-learn/iris", 
    split="train", 
    )

dataset = dataset \
    .map(remove_columns=["Id"]) \
    .rename_columns({
        "SepalLengthCm": "sepal_length", 
        "SepalWidthCm": "sepal_width", 
        "PetalLengthCm": "petal_length",
        "PetalWidthCm": "petal_width",
        "Species": "species"
    })


names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

new_features = Features({
    "petal_length": Value(dtype="float32", id=None),
    "petal_width": Value(dtype="float32", id=None),
    "sepal_length": Value(dtype="float32", id=None),
    "sepal_width": Value(dtype="float32", id=None),
    "species": ClassLabel(names=names)
})

dataset = dataset.cast(new_features)

dataset.push_to_hub("hitorilabs/iris")