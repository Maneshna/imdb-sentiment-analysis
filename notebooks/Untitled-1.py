# %%
import pandas as pd

# %%
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("../data/sentiment.csv")

# %%
df.shape

# %%
df.head

# %%
df.isnull().sum()

# %%
df["sentiment"].value_counts()

# %%
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")


# %%
df["review_length"]=df["review"].apply(len)

# %%
df["review_length"].describe()

# %%
#histogram of review lengths

plt.figure(figsize=(10,5))
plt.hist(df["review_length"], bins=50)

plt.title("Review Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Count")
plt.show()

# %%
df["review"].iloc[0]

# %%
df[df["sentiment"]=="positive"]["review"].iloc[0]

# %%
df[df["sentiment"]=="negative"]["review"].iloc[0]

# %%
df.shape

# %%
#cleaning time 
import re

# %% [markdown]
# Raw Review
# ↓
# Remove HTML
# ↓
# Lowercase
# ↓
# Remove URLs
# ↓
# Remove punctuation
# ↓
# Remove extra spaces
# ↓
# Return Clean Text

# %%
def clean_text(text):
    #convert to lowercase

    text = text.lower()

    #remove the HTML tags

    text=re.sub(r"<.*?>", "", text)

    #remove the urls

    text=re.sub(r"http\S+\www\S+", "", text)

    #remove the punctuation and special characters

    text=re.sub(r"[^a-zA-Z\s]","",text)

    #remove the extra space

    text = re.sub(r"\s+","",text).strip()

    return text

# %%
sample_review = df["review"].iloc[0]

# %%
print("Og")

# %%
print(sample_review)

# %%
sample_review = df["review"].iloc[0]

print("ORIGINAL REVIEW:")
print(sample_review)

print("\n" + "="*100 + "\n")

print("CLEANED REVIEW:")
print(clean_text(sample_review))

# %%
df["cleaned_review"] = df["review"].apply(clean_text)

# %%
print("Original Review")
print(df["review"].iloc[5])

print("\n" + "="*100 + "\n")

print("Cleaned")
print(df["cleaned_review"].iloc[5])

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
vectorizer = TfidfVectorizer(max_features=5000)

# %%
X = vectorizer.fit_transform(df["cleaned_review"])

# %%
X.shape

# %%
from sklearn.preprocessing import LabelEncoder

# %%
encoder = LabelEncoder()
y=encoder.fit_transform(df["sentiment"])

# %%
print(y[:10])

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify=y)

# %%
print(X_train.shape)

# %%
print(X_test.shape)

# %%
from sklearn.linear_model import LogisticRegression

# %%
lr_model = LogisticRegression(max_iter=10000)

# %%
lr_model.fit(X_train,y_train)

# %%
y_pred = lr_model.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
accuracy = accuracy_score(y_test, y_pred)

# %%
print("Accuracy", accuracy)

# %%
print(y_train[:20])
print(y_test[:20])

# %%
import numpy as np
print(np.unique(y_train))
print(np.unique(y_test))

# %%
print(np.unique(y_pred, return_counts=True))

# %%
print(classification_report(y_test, y_pred))

# %%
print(df["cleaned_review"].str.len().describe())

# %%
print(len(vectorizer.vocabulary_))

# %%
print(lr_model.coef_)

# %%
print(lr_model.intercept_)

# %%
train_pred = lr_model.predict(X_train)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, train_pred))

# %%
print(encoder.classes_)
print(np.bincount(y))

# %%
print(lr_model.n_iter_)

# %%
print(type(X_train))
print(type(y_train))

print(X_train.shape)
print(y_train.shape)

# %%
#doing things again

from sklearn.linear_model import LogisticRegression

test_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

test_model.fit(X_train, y_train)

print("n_iter:", test_model.n_iter_)
print("train accuracy:", test_model.score(X_train, y_train))
print("test accuracy:", test_model.score(X_test, y_test))

# %%
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# %%
print(type(X_train))
print(X_train.nnz)

# %%
print(X_train[0])

# %%
print(X_train[0].toarray())

# %%
print(X_train.max())
print(X_train.min())

# %%
print(X_train.sum())

# %%



