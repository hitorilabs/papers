import torch
from config import config
from dataclasses import asdict
import trainer


if __name__ == '__main__':

    with torch.device("cuda:0"):
        model = trainer.ConvNet()
    
    trainloader, testloader = trainer.get_dataloaders(config)

    trainer.train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        config=config, 
        logger_fn=print
        )