import joblib
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix    

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


#to load the path

def load_data(file_path):
    df = pd.read_csv(file_path)
    df["cleaned_review"] = df["review"].apply(clean_text)
    return df



def train_model(df):

    vectorizer = TfidfVectorizer(
        max_features = 10000,
        ngram_range=(1,2))
    X= vectorizer.fit_transform(df["cleaned_review"])
    encoder= LabelEncoder()
    y=encoder.fit_transform(df["sentiment"])
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return (model, vectorizer, encoder, X_train, X_test, y_train, y_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassfication Report:\n")

    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, predictions))




import os

def save_artifacts(
    model,
    vectorizer,
    encoder,
    artifact_dir="artifacts"
):
    os.makedirs(
        artifact_dir,
        exist_ok=True
    )

    joblib.dump(model, os.path.join(artifact_dir, "logistic_model.pkl")
    )

    joblib.dump(vectorizer,os.path.join(artifact_dir,"vectorizer.pkl")
    )

    joblib.dump(encoder,os.path.join(artifact_dir,"label_encoder.pkl")
    )

    print(f"\nArtifacts successfully saved to '{artifact_dir}'")


def main():
    """
    Main training pipeline.
    """

    print("Loading dataset...")

    df = load_data(
        "data/sentiment.csv"
    )

    print("Training model...")

    (
        model,
        vectorizer,
        encoder,
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_model(df)

    print("Evaluating model...")

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print("Saving artifacts...")

    save_artifacts(
        model,
        vectorizer,
        encoder
    )

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()