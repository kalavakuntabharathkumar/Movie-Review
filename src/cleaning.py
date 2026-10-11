import re
import pandas as pd
from bs4 import BeautifulSoup

def clean_review(value: object) -> str:
    if pd.isna(value):
        return ""
    text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"review", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    df = df[["review", "sentiment"]].copy()
    df["review"] = df["review"].map(clean_review)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["review"].ne("") & df["sentiment"].isin(["positive", "negative"])]
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    return df
