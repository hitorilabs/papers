import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent

# 0. Prepare data
df = pd.read_csv(
    current_dir / "iris.data", 
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    )
df = df.loc[df["class"] != "Iris-virginica"]
labels = df.iloc[:,4]

xs = torch.from_numpy(df.iloc[:,:4][["sepal_width", "petal_length"]].to_numpy()).float()

# center data around mean
xs = (xs - xs.mean(dim=0)) / xs.std(dim=0)
ys = torch.tensor(list(map(lambda x: 1 if x == 'Iris-versicolor' else 0, labels)), dtype=torch.long)

# split data into train and test
from sklearn.model_selection import train_test_split
xs, xs_test, ys, ys_test = train_test_split(xs, ys, test_size=0.3, random_state=42)

from torch.utils.data import Dataset, DataLoader
class Data(Dataset):

  def __init__(self, X_train, y_train):
    self.X = X_train
    self.y = y_train
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len

traindata = Data(xs, ys)

# 1. Define the model
class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.sequential = nn.Sequential(
          nn.Linear(input_dim, 25, dtype=torch.float),
          nn.Sigmoid(),
          nn.Linear(25, output_dim, dtype=torch.float),
        )

    def forward(self, x_in):
        x = self.sequential(x_in)
        return x.view(-1)

# 2. Instantiate the model with hyperparameters

model = Perceptron(input_dim=xs.shape[1], output_dim=1)

# 3. Instantiate the loss
criterion = nn.MSELoss()

# 4. Instantiate the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

BATCH_SIZE = 10

trainloader = DataLoader(
    traindata, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=1
    )

# 5. Iterate through the dataset
for epoch in range(20):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Forward pass
        y_pred = model(inputs)

        # Compute Loss
        loss = criterion(y_pred, labels.float())

        # Zero gradients
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

        with torch.no_grad():
            eval_preds = model(xs_test)
            eval_loss = criterion(eval_preds, ys_test.float())

        print(f'[{epoch + 1:3d} {i + 1:3d}] minibatch loss: {loss.item():.5f} eval loss: {eval_loss.item():.5f}')

# 6. Make predictions
with torch.no_grad():
    y_pred = model(xs_test)
    ones = torch.ones(ys_test.size())
    zeros = torch.zeros(ys_test.size())
    test_acc = torch.mean((torch.where(y_pred > 0.5, ones, zeros).int() == ys_test).float())
    print(f"Test accuracy: {test_acc:.5f}")