from typing import Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class CIFAR10Config:
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: int = 0.0068
    num_classes: int = 10
    momentum: float = 0.86583
    weight_decay: float = 0.00834863

config = CIFAR10Config()