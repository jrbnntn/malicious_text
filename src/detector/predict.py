from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.pipeline import Pipeline

from .data import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "baseline_v1.joblib"
SPAM_THRESHOLD = 0.5


class Prediction(TypedDict):
	label: str
	score: float


@lru_cache(maxsize=1)
def _load_model(model_path: Path = MODEL_PATH) -> Pipeline:
	"""Load the saved model once and reuse it for future predictions."""
	return joblib.load(model_path)


def predict(text: str) -> Prediction:
	"""Classify text and return its label and spam probability."""
	if not isinstance(text, str):
		raise TypeError("text must be a string")

	model = _load_model()
	spam_score = float(model.predict_proba([text])[0, 1])
	label = "spam" if spam_score >= SPAM_THRESHOLD else "ham"

	return {"label": label, "score": spam_score}
