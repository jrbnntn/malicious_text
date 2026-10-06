from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_MAPPING = {"ham": 0, "spam": 1}


def load_split(
	split: str,
	project_root: Path = PROJECT_ROOT,
) -> tuple[pd.Series, pd.Series]:
	"""Load a processed split and return its text and numeric labels."""
	split_path = project_root / "data" / "processed" / f"{split}.csv"
	split_df = pd.read_csv(split_path, keep_default_na=False)

	numeric_labels = split_df["label"].map(LABEL_MAPPING)
	if numeric_labels.isna().any():
		unknown_labels = sorted(set(split_df.loc[numeric_labels.isna(), "label"]))
		raise ValueError(f"Unknown labels in {split_path}: {unknown_labels}")

	return split_df["text"], numeric_labels.astype(int)
