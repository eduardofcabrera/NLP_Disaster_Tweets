import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Optimizer
from sentence_transformers import SentenceTransformer


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
        self.loss_list = {"train": [], "val": [], "train_acc": [], "val_acc": []}
        self.save_model_path = save_model_path

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.val_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}, Train Acc={:.5f}, Val Acc={:.5f}, Train F1={:.5f}, Val F1={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss_list["train"][-1],
                    self.loss_list["val"][-1],
                    self.loss_list["train_acc"][-1][0],
                    self.loss_list["val_acc"][-1][0],
                    self.loss_list["train_acc"][-1][1],
                    self.loss_list["val_acc"][-1][1],
                )
            )
        self.save_model()

    def train_epoch(self):
        self.model.train()
        running_loss = []
        y_hat_list = []
        y_list = []
        sigmoid = nn.Sigmoid()

        for batch in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            x = batch[0]
            y = batch[1]
            y_hat = self.model(x)

            output = self.loss(y_hat.squeeze(), y.float())

            y_hat_list.append(sigmoid(y_hat.squeeze()))
            y_list.append(y)

            output.backward()
            self.optimizer.step()

            running_loss.append(output.item())
        y_hat_list = torch.round(torch.concat([y_hat for y_hat in y_hat_list]))
        y_list = torch.concat([y for y in y_list])
        acc = accuracy_score(y_list.tolist(), y_hat_list.tolist())
        f1 = f1_score(y_list.tolist(), y_hat_list.tolist())
        epoch_loss = np.mean(running_loss)
        self.loss_list["train"].append(epoch_loss)
        self.loss_list["train_acc"].append((acc, f1))

    def val_epoch(self):
        self.model.eval()
        running_loss = []
        y_hat_list = []
        y_list = []

        sigmoid = nn.Sigmoid()

        for batch in tqdm(self.val_dataloader):
            x = batch[0]
            y = batch[1]
            y_hat = self.model(x)
            output = self.loss(y_hat.squeeze(), y.float())

            y_hat_list.append(sigmoid(y_hat.squeeze()))
            y_list.append(y)

            running_loss.append(output.item())
        y_hat_list = torch.round(torch.concat([y_hat for y_hat in y_hat_list]))
        y_list = torch.concat([y for y in y_list])
        acc = accuracy_score(y_list.tolist(), y_hat_list.tolist())
        f1 = f1_score(y_list.tolist(), y_hat_list.tolist())
        epoch_loss = np.mean(running_loss)
        self.loss_list["val"].append(epoch_loss)
        self.loss_list["val_acc"].append((acc, f1))

    def save_model(self):
        torch.save(self.model, self.save_model_path)


class Inference:
    def __init__(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        sample_submission_path: str,
        save_result_path: str,
    ):
        self.model = model
        self.test_dataloader = test_dataloader
        self.save_result_path = save_result_path
        self.df_submission = pd.read_csv(sample_submission_path)

    def inference(self):
        self.model.eval()
        sigmoid = nn.Sigmoid()
        y_hat_list = []

        for batch in tqdm(self.test_dataloader):
            y_hat = self.model(batch)
            y_hat_list.append(sigmoid(y_hat.squeeze()))
        y_hat_list = torch.round(torch.concat([y_hat for y_hat in y_hat_list]))
        self.df_submission["target"] = y_hat_list.int().tolist()

        result_df = self.df_submission[["id", "target"]]
        result_df = result_df.set_index("id")

        result_df.to_csv(self.save_result_path)
