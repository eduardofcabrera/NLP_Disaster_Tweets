import pandas as pd
import torchtext
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class TextCBOWDataset(Dataset):
    def __init__(
        self,
        csv_file_path: str,
        vocab: torchtext.vocab.Vocab,
        tokenizer,
        context_size: int,
    ) -> None:
        df: pd.DataFrame = pd.read_csv(csv_file_path)[["text", "target"]]
        self.vocab: torchtext.vocab.Vocab = vocab
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.cbow_list = self.generate_cbow_list(
            text_list=df["text"].tolist(), context_size=context_size
        )
        self.vocab_size = len(vocab)

    def __len__(self):
        return len(self.cbow_list)

    def generate_cbow_list(
        self, text_list: list[str], context_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cbow_list = []
        for tokens in tqdm(
            list(map(self.vocab.lookup_indices, map(self.tokenizer, text_list)))
        ):
            tokens = torch.tensor(tokens)
            if len(tokens) < context_size * 2 + 1:
                continue
            n_times = len(tokens) - context_size * 2
            for n in range(n_times):
                context_left = tokens[n : n + context_size]
                context_right = tokens[n + context_size + 1 : n + context_size + 3]
                token_mid = tokens[n + context_size]
                cbow_list.append(
                    (torch.concat([context_left, context_right]), token_mid)
                )

        return cbow_list

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        cbow_value = self.cbow_list[idx]
        target = cbow_value[1]
        one_hot_target = torch.zeros(self.vocab_size)
        one_hot_target[target] = 1

        return cbow_value[0], target  # one_hot_target


class TextClassificationDataset(Dataset):
    def __init__(
        self, csv_file_path: str, vocab: torchtext.vocab.Vocab, tokenizer
    ) -> None:
        df: pd.DataFrame = pd.read_csv(csv_file_path)[
            ["text", "keyword", "location", "target"]
        ]

        self.sentences = self.get_padded_sequence(df["text"], vocab, tokenizer)
        self.locations = self.get_padded_sequence(
            df["location"].fillna("<na>"), vocab, tokenizer
        )
        self.keywords = self.get_padded_sequence(
            df["keyword"].fillna("<na>"), vocab, tokenizer
        )
        self.targets = df["target"]
        self.vocab: torchtext.vocab.Vocab = vocab
        self.tokenizer = tokenizer

    def get_padded_sequence(self, sequences, vocab, tokenizer):
        sentences = [
            torch.tensor(sentence)
            for sentence in map(vocab.lookup_indices, map(tokenizer, sequences))
        ]
        max_len = max([len(sentence) for sentence in sentences])
        padded_sentences = [
            nn.functional.pad(
                sentence,
                (max_len - len(sentence), 0),
                "constant",
            )
            for sentence in sentences
        ]
        return padded_sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = (self.sentences[idx], self.keywords[idx], self.locations[idx])
        y = torch.tensor(self.targets[idx])

        return x, y


class TextClassificationDatasetInference(Dataset):
    def __init__(
        self, csv_file_path: str, vocab: torchtext.vocab.Vocab, tokenizer
    ) -> None:
        df: pd.DataFrame = pd.read_csv(csv_file_path)[["text", "keyword", "location"]]

        self.sentences = self.get_padded_sequence(df["text"], vocab, tokenizer)
        self.locations = self.get_padded_sequence(
            df["location"].fillna("<na>"), vocab, tokenizer
        )
        self.keywords = self.get_padded_sequence(
            df["keyword"].fillna("<na>"), vocab, tokenizer
        )
        self.vocab: torchtext.vocab.Vocab = vocab
        self.tokenizer = tokenizer

    def get_padded_sequence(self, sequences, vocab, tokenizer):
        sentences = [
            torch.tensor(sentence)
            for sentence in map(vocab.lookup_indices, map(tokenizer, sequences))
        ]
        max_len = max([len(sentence) for sentence in sentences])
        padded_sentences = [
            nn.functional.pad(
                sentence,
                (max_len - len(sentence), 0),
                "constant",
            )
            for sentence in sentences
        ]
        return padded_sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = (self.sentences[idx], self.keywords[idx], self.locations[idx])

        return x
