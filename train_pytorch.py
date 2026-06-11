import re
import os
import pandas as pd
import torch
import torch.nn as nn

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["cleaned_review"] = df["review"].apply(clean_text)
    return df


def build_vocab(texts, max_vocab_size=20000):
    counter = Counter()

    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    most_common_words = counter.most_common(max_vocab_size - 2)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for idx, (word, _) in enumerate(most_common_words, start=2):
        vocab[word] = idx

    return vocab


def encode_text(text, vocab):
    tokens = text.split()

    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]


def pad_sequence(sequence, max_length, pad_idx=0):
    if len(sequence) > max_length:
        return sequence[:max_length]

    return sequence + [pad_idx] * (max_length - len(sequence))


class SentimentDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            embed_dim,
            64
        )
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(
            64,
            2)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        pooled = embedded.mean(dim=1)
        output = self.fc(pooled)
        return output


def main():

    df = load_data("data/sentiment.csv")

    vocab = build_vocab(df["cleaned_review"])

    print("Vocabulary Size:", len(vocab))

    MAX_LENGTH = 200

    texts = [
        pad_sequence(
            encode_text(text, vocab),
            MAX_LENGTH
        )
        for text in df["cleaned_review"]
    ]

    labels = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    }).tolist()

    X = torch.tensor(
        texts,
        dtype=torch.long
    )

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    print("X Shape:", X.shape)
    print("y Shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_dataset = SentimentDataset(
        X_train,
        y_train
    )

    test_dataset = SentimentDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using:", device)

    model = SentimentModel(
        vocab_size=len(vocab),
        embed_dim=128
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    EPOCHS = 10

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

        accuracy = (
            correct / total
        ) * 100

        avg_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

    test_accuracy = (
        correct / total
    ) * 100

    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "artifacts/pytorch_model.pt"
    )
    import joblib


    joblib.dump(
        vocab,
        "artifacts/vocab.pkl"
    )

    print("\nPyTorch artifacts saved.")


if __name__ == "__main__":
    main()

