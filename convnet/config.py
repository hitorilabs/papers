from typing import Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class CIFAR10Config:
    batch_size: int = 3
    num_epochs: int = 50
    learning_rate: int = 1e-3
    num_classes: int = 10
    data_shape: Tuple[int, int, int]= (3, 32, 32) # (channels, height, width)
    momentum: float = 0.0
    weight_decay: float = 0.0

config = CIFAR10Config()