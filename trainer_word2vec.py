import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Optimizer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        loss: nn.Module,
        epochs: int,
        save_model_path: str,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.loss_list = {"train": [], "val": []}
        self.save_model_path = save_model_path

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.val_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss_list["train"][-1],
                    self.loss_list["val"][-1],
                )
            )
        self.save_model()

    def train_epoch(self):
        self.model.train()
        running_loss = []

        for batch in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            x = batch[0]
            y = batch[1]
            y_hat = self.model(x)

            output = self.loss(y_hat, y)

            output.backward()
            self.optimizer.step()

            running_loss.append(output.item())
        epoch_loss = np.mean(running_loss)
        self.loss_list["train"].append(epoch_loss)

    def val_epoch(self):
        self.model.eval()
        running_loss = []

        for batch in tqdm(self.val_dataloader):
            x = batch[0]
            y = batch[1]
            y_hat = self.model(x)
            output = self.loss(y_hat, y)

            running_loss.append(output.item())
        epoch_loss = np.mean(running_loss)
        self.loss_list["val"].append(epoch_loss)

    def save_model(self):
        torch.save(self.model, self.save_model_path)
