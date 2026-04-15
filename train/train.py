import torch
import torch.nn as nn
import torch.optim as optim

from model.swift_transformer import SwiftTransformer
from utils.tokenizer import SimpleTokenizer

# Load data
text = open("data/sample.txt").read()

tokenizer = SimpleTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Hyperparams
vocab_size = len(tokenizer.stoi)
model = SwiftTransformer(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    inputs = data[:-1].unsqueeze(0)
    targets = data[1:].unsqueeze(0)

    outputs = model(inputs)
    loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
