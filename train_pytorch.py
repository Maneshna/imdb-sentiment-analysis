import re
import pandas as pd
import torch
from collections import Counter

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_data(file_path):
    df=pd.read_csv(file_path)
    df["cleaned_review"]=df["review"].apply(clean_text)
    return df


#build vocabulary
def build_vocab(texts, max_vocab_size=20000):

    counter = Counter()
    for text in texts:

        tokens=text.split()
        counter.update(tokens)
    most_common_words = counter.most_common(max_vocab_size - 2)
    vocab = {"<PAD>":0,
             "<UNK>":1}
    for idx, (word,_) in enumerate(most_common_words, start=2):
        vocab[word]=idx
    return vocab

df=load_data("data/sentiment.csv")
vocab=build_vocab(df["cleaned_review"])



#IDs
def encode_text(text, vocab):
    tokens = text.split()
    encoded = [vocab.get(token, vocab["<UNK>"]) 
              for token in tokens]
    return encoded

sample = df["cleaned_review"].iloc[0]

encoded = encode_text(
    sample, vocab)



def pad_sequence(sequence, max_length, pad_idx=0):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_idx] * (max_length - len(sequence))

sample_ids = encode_text(sample, vocab)

padded = pad_sequence(sample_ids, max_length=20)
                      


MAX_LENGTH=200

texts = [pad_sequence(encode_text(text, vocab), MAX_LENGTH)
        for text in df["cleaned_review"]]
labels = df["sentiment"].map({"negative":0,"positive":1}).tolist()

#convert into tensors

X=torch.tensor(texts,dtype=torch.long)
y=torch.tensor(labels, dtype=torch.long)



#train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
#create the dataloaders

train_dataset = SentimentDataset(X_train, y_train)

test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

import torch.nn as nn
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)
        self.fc =nn.Linear(embed_dim, 2)

    def forward(self, x):
        embedded=self.embedding(x)
        pooled=embedded.mean(dim=1)
        output = self.fc(pooled)
        return output


#initialize the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentModel(vocab_size=len(vocab)).to(device)
    
print(model)
print(device)








