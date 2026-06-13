# Sentiment Analysis Web Application

An end-to-end sentiment analysis web application built using Machine Learning and Deep Learning models.

The application allows users to enter movie reviews and predict whether the sentiment is positive or negative using either:

- Logistic Regression (Scikit-Learn)
- PyTorch Neural Network

The project includes a FastAPI backend, Streamlit frontend, model training pipelines, and deployed inference APIs.

---

## Features

- Text preprocessing and cleaning
- TF-IDF Vectorization
- Logistic Regression sentiment classifier
- PyTorch Neural Network sentiment classifier
- FastAPI REST API
- Streamlit Web Interface
- Confidence score prediction
- Dual-model selection (ML and Deep Learning)

---

## Project Structure

```text
sentiment-analysis/
│
├── artifacts/
│   ├── logistic_model.pkl
│   ├── vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── pytorch_model.pt
│   └── vocab.pkl
│
├── data/
│   └── sentiment.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── train.py
├── train_pytorch.py
├── app_backend.py
├── app_frontend.py
├── requirements.txt
└── README.md
```

---

## Models Used

### Logistic Regression

- TF-IDF Vectorization
- Scikit-Learn Logistic Regression
- Fast and lightweight baseline model

### PyTorch Neural Network

Architecture:

```text
Embedding Layer
        ↓
Mean Pooling
        ↓
Linear (128 → 64)
        ↓
ReLU
        ↓
Dropout
        ↓
Linear (64 → 2)
```

---

## Dataset

IMDb Movie Reviews Dataset

- 50,000 movie reviews
- Balanced positive and negative sentiments

---

## Model Performance

| Model | Test Accuracy |
|---------|---------|
| Logistic Regression | ~89% |
| PyTorch Neural Network | ~88% |

Performance may vary slightly depending on training runs.

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_LINK_HERE
cd sentiment-analysis
```

Create environment:

```bash
conda create -n torch_env python=3.11
conda activate torch_env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Models

### Logistic Regression

```bash
python train.py
```

### PyTorch Model

```bash
python train_pytorch.py
```

Artifacts will be saved automatically in:

```text
artifacts/
```

---

## Run FastAPI Backend

```bash
uvicorn app_backend:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run Streamlit Frontend

```bash
streamlit run app_frontend.py
```

---

## Example API Request

```json
{
    "text": "This movie was amazing. I loved it.",
    "model_type": "pytorch"
}
```

Example Response:

```json
{
    "sentiment": "positive",
    "confidence": 0.93
}
```

---

## Technologies Used

- Python
- Scikit-Learn
- PyTorch
- FastAPI
- Streamlit
- Pandas
- NumPy

---

## Author

Maneshna

GitHub:
[Github Profile](https://github.com/Maneshna)

Project Repository:
[Project Link](https://github.com/Maneshna/imdb-sentiment-analysis)