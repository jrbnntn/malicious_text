import json
from pathlib import Path
from datetime import date

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .data import PROJECT_ROOT, load_split


MODEL_PATH = PROJECT_ROOT / "models" / "baseline_v1.joblib"
METADATA_PATH = MODEL_PATH.with_suffix(".json")
RANDOM_SEED = 42
NGRAM_RANGE = (1, 1)
CLASS_WEIGHT = None
SPAM_THRESHOLD = 0.5


def build_pipeline() -> Pipeline:
    """Build the baseline unigram TF-IDF and logistic regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=NGRAM_RANGE,
            min_df=2,
        )),
        ("logistic_regression", LogisticRegression(
            max_iter=1000,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_SEED,
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

    validation_probabilities = pipeline.predict_proba(validation_text)[:, 1]
    validation_predictions = (validation_probabilities >= SPAM_THRESHOLD).astype(int)
    metrics = {
        "accuracy": accuracy_score(validation_labels, validation_predictions),
        "spam_f1": f1_score(validation_labels, validation_predictions, pos_label=1),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata = {
        "random_seed": RANDOM_SEED,
        "ngram_range": list(NGRAM_RANGE),
        "class_weight": CLASS_WEIGHT,
        "threshold": SPAM_THRESHOLD,
        "validation_metrics": metrics,
        "date": date.today().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "security_note": "Pickled models are unsafe to load from untrusted sources.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return pipeline, metrics


def main() -> None:
    _, metrics = train_and_evaluate()
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation spam F1: {metrics['spam_f1']:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
