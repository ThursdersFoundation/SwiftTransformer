import torch
import torch.nn as nn

class SwiftTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x)
