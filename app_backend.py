from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import re

import torch
import torch.nn as nn


app = FastAPI(
    title="Sentiment Analysis API"
)

MAX_LENGTH = 200

# Load Logistic Regression Artifacts


model = joblib.load(
    "artifacts/logistic_model.pkl"
)

vectorizer = joblib.load("artifacts/vectorizer.pkl")

encoder = joblib.load("artifacts/label_encoder.pkl")

# Load PyTorch Artifacts

vocab = joblib.load("artifacts/vocab.pkl")

# Text Cleaning

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>","",text)

    text = re.sub(r"http\S+|www\S+","",text)

    text = re.sub(r"[^a-zA-Z\s]","",text)

    text = re.sub(r"\s+"," ",text).strip()

    return text

# Request / Response Models


class PredictionRequest(BaseModel):
    text: str
    model_type: str = "logistic_regression"


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

# PyTorch Model


class SentimentModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(embed_dim,64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(64,2)

    def forward(self, x):

        embedded = self.embedding(x)

        embedded = self.dropout(
            embedded
        )

        pooled = embedded.mean(dim=1)

        x = self.fc(pooled)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x
    

pytorch_model = SentimentModel(
    vocab_size=len(vocab),
    embed_dim=128
)

pytorch_model.load_state_dict(
    torch.load(
        "artifacts/pytorch_model.pt",
        map_location="cpu"
    )
)

pytorch_model.eval()

# PyTorch Utilities


def encode_text(text,vocab):
    tokens = text.split()

    return [
        vocab.get(token,vocab["<UNK>"])
        for token in tokens
    ]


def pad_sequence(
    sequence,
    max_length,
    pad_idx=0
):

    if len(sequence) > max_length:

        return sequence[:max_length]

    return sequence + (
        [pad_idx] *
        (max_length - len(sequence))
    )


def predict_pytorch(text):

    encoded = encode_text(text,vocab)

    padded = pad_sequence(encoded,MAX_LENGTH)

    tensor = torch.tensor([padded],dtype=torch.long)

    

    with torch.no_grad():

        outputs = pytorch_model(tensor)
        print("Output Shape:", outputs.shape)
        probabilities = torch.softmax(outputs,dim=1)

        prediction = torch.argmax(probabilities,dim=1).item()

        confidence = probabilities.max().item()
        print("Outputs:", outputs)
        print("Probabilities:", probabilities)
        print("Prediction:", prediction)
        print("Confidence:", confidence)

    sentiment = ("positive"
        if prediction == 1
        else "negative"
    )

    return (
        sentiment,
        confidence
    )


@app.get("/")
def home():

    return {
        "message":
        "Sentiment Analysis API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_sentiment(
    request: PredictionRequest
):

    cleaned_text = clean_text(request.text)

    # PyTorch Prediction
    

    if request.model_type == "pytorch":

        sentiment, confidence = (
            predict_pytorch(cleaned_text))
  
    # Logistic Regression
    
    else:

        vector = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            vector
        )[0]

        probabilities = (
            model.predict_proba(
                vector
            )[0]
        )

        sentiment = (encoder.inverse_transform([prediction])[0])

        confidence = float(
            max(probabilities)
        )

    return PredictionResponse(
        sentiment=sentiment,
        confidence=confidence
    )