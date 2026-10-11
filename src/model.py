from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=100_000,
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
