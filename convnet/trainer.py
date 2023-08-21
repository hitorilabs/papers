from typing import Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ) 
        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv_pool1(x)  # (3, 32, 32) => (6, 28, 28) => (6, 14, 14)
        x = self.conv_pool2(x)  # (6, 14, 14) => (16, 10, 10) => (16, 5, 5)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)         # (16, 5, 5) => (16 * 5 * 5) => (120)
        x = self.fc2(x)         # (120) => (84)
        x = self.fc3(x)         # (84) => (10)
        return x

def train(model, trainloader, testloader, config, logger_fn):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.learning_rate, 
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        )

    for _ in range(config.num_epochs):

        start = time.monotonic()
        running_loss = 0.
        last_loss = 0.
        for idx, [data, targets] in enumerate(trainloader):
            optimizer.zero_grad()
            loss = criterion(model(data.to(device, non_blocking=True)), targets.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if idx % 1000 == 999:
                last_loss = running_loss / 1000 # loss per mini-batch
                running_loss = 0.0
        
        end = time.monotonic() 

        with torch.no_grad():
            correct = 0
            total = 0

            for data, targets in testloader:
                outputs = model(data.to(device, non_blocking=True))
                pred = torch.argmax(outputs, 1)
                total += pred.size(0)
                correct += (pred == targets.to(device, non_blocking=True)).sum().item()

            logger_fn({"val_accuracy": correct / total, "train_loss": last_loss, "epoch_time": end - start})

def get_dataloaders(config) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                            shuffle=True, num_workers=2, 
                                            pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                            shuffle=False, num_workers=2,
                                            pin_memory=True)
    
    return trainloader, testloader