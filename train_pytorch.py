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
print("Vocabulary Size:", len(vocab))
print(list(vocab.items())[:20])


#IDs
def encode_text(text, vocab):
    tokens = text.split()
    encoded = [vocab.get(token, vocab["<UNK>"]) 
              for token in tokens]
    return encoded

sample = df["cleaned_review"].iloc[0]
print(sample[:100])
encoded = encode_text(
    sample, vocab)

print(encoded[:20])

def pad_sequence(sequence, max_length, pad_idx=0):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_idx] * (max_length - len(sequence))

sample_ids = encode_text(sample, vocab)

padded = pad_sequence(sample_ids, max_length=20)
                      
print(padded)
print(len(padded))


