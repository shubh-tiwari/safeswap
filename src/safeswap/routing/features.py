"""Request features for learned gates. Text only: no task/segment labels, since live traffic has none.

* tfidf: word + char n-gram TF-IDF (scikit-learn). Nothing to download; fast.
* embed: sentence-transformer embeddings, cached to .cache/emb-<model>-<hash>.npy. The model is
  fetched from the HF Hub anonymously (token=False) the first time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Tfidf:
    def __init__(self, max_features: int = 50_000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.vec = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )

    def _x(self, texts: pd.Series):
        lengths = np.log1p(texts.str.len().to_numpy(float))[:, None] / 10
        return lengths

    def fit_transform(self, texts: pd.Series):
        return sparse.hstack([self.vec.fit_transform(texts), self._x(texts)]).tocsr()

    def transform(self, texts: pd.Series):
        return sparse.hstack([self.vec.transform(texts), self._x(texts)]).tocsr()


class Embed:
    def __init__(self, model: str = EMBED_MODEL, cache_dir: str | Path = ".cache"):
        self.model_name = model
        self.cache_dir = Path(cache_dir)
        self._model = None

    def _encode(self, texts: pd.Series) -> np.ndarray:
        h = hashlib.sha256(("\x00".join(texts) + self.model_name).encode()).hexdigest()[:16]
        path = self.cache_dir / f"emb-{self.model_name.split('/')[-1]}-{h}.npy"
        if path.exists():
            return np.load(path)
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, token=False)
        x = self._model.encode(
            texts.str.slice(0, 2000).tolist(),
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, x)
        return x

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        return self._encode(texts)

    def transform(self, texts: pd.Series) -> np.ndarray:
        return self._encode(texts)


def make(kind: str):
    if kind == "tfidf":
        return Tfidf()
    if kind == "embed":
        return Embed()
    raise ValueError(f"unknown features: {kind}")
