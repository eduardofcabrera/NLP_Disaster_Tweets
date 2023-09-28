import pandas as pd
import torchtext
from torchtext.data import get_tokenizer
import torch.nn as nn
import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse
import yaml

from model import CBOW_Model
from dataset import TextCBOWDataset
from trainer_word2vec import Trainer


def build_vocab(
    csv_file_path: str, tokenizer, min_freq: int, save_vocab_filepath: str
) -> torchtext.vocab.Vocab:
    train_df = pd.read_csv(csv_file_path)
    text = train_df["text"]
    keyword = train_df["keyword"]
    location = train_df["location"]
    data_iter = pd.concat([text, keyword, location])
    data_iter = data_iter.dropna()
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter), specials=["<unk>"], min_freq=min_freq
    )
    vocab.set_default_index(vocab["<unk>"])
    torch.save(vocab, save_vocab_filepath)
    return vocab


def word2vec(config: dict[str:any]):
    train_csv = config["train_csv"]
    val_csv = config["val_csv"]
    test_csv = config["test_csv"]
    word2vec_csv = config["word2vec_csv"]
    save_vocab = config["save_vocab"]
    save_cbow = config["save_cbow"]
    min_freq = config["min_freq"]
    context_size = config["context_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    max_norm = config["max_norm"]
    embedding_dim = config["embedding_dim"]

    tokenizer = get_tokenizer(config["tokenizer"])
    vocab = build_vocab(word2vec_csv, tokenizer, min_freq, save_vocab)

    cbow_ds_train = TextCBOWDataset(word2vec_csv, vocab, tokenizer, context_size)
    cbow_ds_val = TextCBOWDataset(val_csv, vocab, tokenizer, context_size)

    train_dataloader = DataLoader(cbow_ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(cbow_ds_val, batch_size=batch_size, shuffle=True)
    model = CBOW_Model(len(vocab), embedding_dim, max_norm)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss,
        epochs,
        save_cbow,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path to yaml config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    word2vec(config)
