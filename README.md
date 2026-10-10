# Movie Review Sentiment Analyzer

A reproducible text-classification project using pandas, scikit-learn TF-IDF, Logistic Regression, and Matplotlib.

## Dataset
Download the Kaggle **IMDB Dataset of 50K Movie Reviews** CSV and place it at `data/IMDB Dataset.csv`. Expected columns: `review` and `sentiment`. The dataset is not bundled because it is distributed separately by Kaggle and may have its own access/license conditions. Do not commit the dataset to Git.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python train.py
python inspect_errors.py
```
The training script removes duplicate reviews, strips HTML tags, reports class balance, creates an 80/20 stratified split, trains TF-IDF + Logistic Regression, prints test metrics, saves the model, and plots the strongest positive/negative coefficients. Results are computed from the CSV supplied by you; no accuracy result is hard-coded. The error-inspection script prints up to 50 misclassified reviews so patterns such as sarcasm and mixed opinions can be checked manually.
