import torch.nn as nn
import torch


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_norm: int) -> None:
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, max_norm=max_norm
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, inputs_: torch.Tensor):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_norm: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(SequenceEncoder, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, max_norm=max_norm
        )

        self.drop_1 = nn.Dropout(p=0.5)

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )

        self.norm = nn.BatchNorm1d(num_features=hidden_size)

        self.drop_2 = nn.Dropout(p=0.5)

    def forward(self, inputs_: torch.tensor):
        text = inputs_[0]
        keywords = inputs_[1]
        locations = inputs_[2]

        text = self.embeddings(text.int())
        keywords = self.embeddings(keywords.int())
        locations = self.embeddings(locations.int())

        text = self.drop_1(text)
        keywords = self.drop_1(keywords)
        locations = self.drop_1(locations)

        text = self.rnn(text)[-1][-1]
        keywords = self.rnn(keywords)[-1][-1]
        locations = self.rnn(locations)[-1][-1]

        text = self.drop_2(text)
        keywords = self.drop_2(keywords)
        locations = self.drop_2(locations)

        x = torch.concat([text, keywords, locations], axis=-1)

        return x


class SequenceDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super(SequenceDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size * 3, out_features=hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_size, out_features=output_dim),
        )

    def forward(self, inputs_):
        x = self.mlp(inputs_)

        return x


class TextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_norm: int,
        hidden_size: int,
        num_layers: int,
        output_dim: int,
    ):
        super(TextClassificationModel, self).__init__()

        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.decoder = SequenceDecoder(hidden_size=hidden_size, output_dim=output_dim)

    def forward(self, inputs_):
        x = self.encoder(inputs_)
        x = self.decoder(x)

        return x
