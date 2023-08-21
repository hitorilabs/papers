import torch
import wandb
from config import config
from dataclasses import asdict
import trainer


if __name__ == '__main__':

    run = wandb.init(
        project="cifar10",
        config=asdict(config)
    )

    with torch.device("cuda:0"):
        model = trainer.ConvNet()
    
    wandb.watch(model)

    trainloader, testloader = trainer.get_dataloaders(config)

    trainer.train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        config=wandb.config, 
        logger_fn=wandb.log
        )