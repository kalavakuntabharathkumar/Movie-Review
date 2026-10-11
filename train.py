from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.cleaning import load_and_clean
from src.model import build_model
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "IMDB Dataset.csv"
OUT = ROOT / "outputs"

def plot_top_words(model, output_path: Path, n: int = 10):
    vectorizer = model.named_steps["tfidf"]
    weights = model.named_steps["classifier"].coef_[0]
    names = vectorizer.get_feature_names_out()
    pos, neg = weights.argsort()[-n:][::-1], weights.argsort()[:n]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(names[pos][::-1], weights[pos][::-1]); axes[0].set_title("Strongest positive features")
    axes[1].barh(names[neg], weights[neg]); axes[1].set_title("Strongest negative features")
    fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)

def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}\nDownload the Kaggle IMDB 50K CSV and place it at data/IMDB Dataset.csv.")
    OUT.mkdir(exist_ok=True)
    df = load_and_clean(str(DATA_PATH))
    if df["label"].nunique() != 2 or len(df) < 10:
        raise SystemExit("Need at least 10 cleaned reviews containing both sentiment classes.")
    print(f"Cleaned unique reviews: {len(df):,}\n")
    print(df["sentiment"].value_counts())
    X_train, X_test, y_train, y_test = train_test_split(
        df.review, df.label, test_size=0.20, random_state=42, stratify=df.label)
    model = build_model(); model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\nMajority-class baseline: {y_test.value_counts(normalize=True).max():.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["negative", "positive"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    joblib.dump(model, OUT / "sentiment_pipeline.joblib")
    plot_top_words(model, OUT / "top_sentiment_words.png")
    print(f"Saved model and chart in {OUT}")

if __name__ == "__main__": main()
