from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(
    title="Sentiment Analysis API",
)
model = joblib.load("artifacts/logistic_model.pkl")

vectorizer = joblib.load("artifacts/vectorizer.pkl")

encoder = joblib.load("artifacts/label_encoder.pkl")
 

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

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running"
    }

class PredictionRespose(BaseModel):
    sentiment: str
    confidence: float

@app.post(

    "/predict",
    response_model=PredictionRespose
)

def predict_sentiment(request: PredictionRequest):

    cleaned_text = clean_text(
        request.text
    )

    vector = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    sentiment = encoder.inverse_transform([prediction])[0]

    confidence = float(max(probabilities))

    return PredictionRespose(sentiment=sentiment, confidence=confidence)


