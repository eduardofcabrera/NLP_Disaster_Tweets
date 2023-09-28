import pandas as pd
import torchtext
from torchtext.data import get_tokenizer
import torch.nn as nn
import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import CBOW_Model, TextClassificationModel
from dataset import (
    TextCBOWDataset,
    TextClassificationDataset,
    TextClassificationDatasetInference,
)
from trainer_text_classification import Trainer, Inference
import argparse
import yaml


def textClassification(config: dict[str:any]):
    train_csv = config["train_csv"]
    val_csv = config["val_csv"]
    save_vocab = config["save_vocab"]
    save_cbow = config["save_cbow"]
    save_model = config["save_model"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    max_norm = config["max_norm"]
    embedding_dim = config["embedding_dim"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    output_dim = config["output_dim"]
    freeze = config["freeze"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cbow_model = torch.load(save_cbow, map_location=device)
    vocab = torch.load(save_vocab)

    tokenizer = get_tokenizer(config["tokenizer"])

    vocab_size = len(vocab)

    train_ds = TextClassificationDataset(train_csv, vocab, tokenizer)
    val_ds = TextClassificationDataset(val_csv, vocab, tokenizer)

    model = TextClassificationModel(
        vocab_size, embedding_dim, max_norm, hidden_size, num_layers, output_dim
    )

    model.embeddings = cbow_model.embeddings

    if freeze:
        for param in model.embeddings.parameters():
            param.requires_grad = False

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss,
        epochs,
        save_model,
    )
    trainer.train()


def textClassificationInference(config: dict[str:any]):
    test_csv = config["test_csv"]
    submission_csv = config["submission_csv"]
    save_submission_csv = config["save_submission_csv"]
    save_vocab = config["save_vocab"]
    save_model = config["save_model"]
    batch_size = config["batch_size"]

    model = torch.load(save_model)
    vocab = torch.load(save_vocab)

    tokenizer = get_tokenizer(config["tokenizer"])

    test_ds = TextClassificationDatasetInference(test_csv, vocab, tokenizer)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    inference = Inference(model, test_dataloader, submission_csv, save_submission_csv)
    inference.inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path to yaml config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config["inference"]:
        textClassificationInference(config)
    else:
        textClassification(config)
