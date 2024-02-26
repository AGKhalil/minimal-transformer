from typing import Any
from typing import Optional

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: Optional[Module] = None,
        device: str = "cuda",
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[_Loss] = None,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        logger: Any = None,
    ):
        """Initialize the Trainer with the given parameters.

        Args:
            model (Optional[Module]): The model to be trained.
            device (str): The device to use for training ('cuda' or 'cpu').
            optimizer (Optional[Optimizer]): The optimizer for training.
            criterion (Optional[_Loss]): The loss function.
            train_loader (Optional[DataLoader]): DataLoader for the training
                data.
            test_loader (Optional[DataLoader]): DataLoader for the testing
                data.
            num_epochs (int): The number of epochs to train for.
            logger (Any): The logger used to track progress.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.logger = logger

    def fit(self):
        """Train the model for a specified number of epochs."""
        epochs = tqdm(range(self.num_epochs), desc="Epochs")
        training_progress = tqdm(
            total=len(self.train_loader) - 1, desc="Training progress"
        )

        for epoch in epochs:
            training_progress.reset()
            for x, y in self.train_loader:
                training_progress.update()
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x, y)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), y.view(-1)
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.logger.log({"train/loss": loss.item(), "epoch": epoch})
