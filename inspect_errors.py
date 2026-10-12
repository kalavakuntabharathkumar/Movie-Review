from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.cleaning import load_and_clean
from src.model import build_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "IMDB Dataset.csv"

def main():
    if not DATA_PATH.exists(): raise SystemExit("Place Kaggle CSV at data/IMDB Dataset.csv first.")
    df = load_and_clean(str(DATA_PATH))
    Xtr, Xte, ytr, yte = train_test_split(df.review, df.label, test_size=.20, random_state=42, stratify=df.label)
    model = build_model(); model.fit(Xtr, ytr); pred = model.predict(Xte)
    errors = pd.DataFrame({"review": Xte.reset_index(drop=True),
        "actual": yte.reset_index(drop=True).map({0:"negative",1:"positive"}),
        "predicted": pd.Series(pred).map({0:"negative",1:"positive"})})
    errors = errors[errors.actual != errors.predicted].head(50)
    print(f"Showing {len(errors)} misclassified reviews:")
    for i, row in errors.iterrows():
        print(f"\n--- Example {i+1} | actual={row.actual} predicted={row.predicted} ---\n{row.review[:1200]}")

if __name__ == "__main__": main()
