import os
import glob
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

DATA_DIR = os.path.join("data", "sst2")


def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=(ext == ".jsonl"))
    raise ValueError(f"Unsupported file type: {path}")


def load_split(keyword):
    files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*"))
        if keyword in os.path.basename(f).lower()
        and "unsup" not in os.path.basename(f).lower()
    ]
    if not files:
        raise FileNotFoundError(f"No file containing '{keyword}' found in {DATA_DIR}/")
    print(f"Loading {keyword} data from: {files[0]}")
    df = read_file(files[0])
    print("Columns found:", list(df.columns))

    # Find the text and label columns
    text_col = next(c for c in df.columns if c.lower() in ("text", "review", "sentence"))
    label_col = next(c for c in df.columns if c.lower() in ("label", "sentiment"))

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    # Convert "positive"/"negative" strings to 1/0 if needed
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"positive": 1, "negative": 0})

    df = df[df["label"].isin([0, 1])]   # drop unlabeled rows (-1)
    df["label"] = df["label"].astype(int)
    return df


train = load_split("train")
test = load_split("test")
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Build the model
model = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1, 2), stop_words="english"),
    LogisticRegression(max_iter=1000),
)

print("Training model...")
model.fit(train["text"], train["label"])

print("Evaluating...")
preds = model.predict(test["text"])
print(classification_report(test["label"], preds, target_names=["Negative", "Positive"]))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("Model saved to models/model.joblib")