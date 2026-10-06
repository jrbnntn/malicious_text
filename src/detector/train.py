from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .data import PROJECT_ROOT, load_split


MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_pipeline.joblib"


def build_pipeline() -> Pipeline:
    """Build the baseline unigram TF-IDF and logistic regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            min_df=2,
        )),
        ("logistic_regression", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])


def train_and_evaluate(
    model_path: Path = MODEL_PATH,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit on the training split, evaluate on validation, and save the model."""
    train_text, train_labels = load_split("train")
    validation_text, validation_labels = load_split("validation")

    pipeline = build_pipeline()
    pipeline.fit(train_text, train_labels)

    validation_predictions = pipeline.predict(validation_text)
    metrics = {
        "accuracy": accuracy_score(validation_labels, validation_predictions),
        "spam_f1": f1_score(validation_labels, validation_predictions, pos_label=1),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline, metrics


def main() -> None:
    _, metrics = train_and_evaluate()
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation spam F1: {metrics['spam_f1']:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
