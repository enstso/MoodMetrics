from pathlib import Path

import joblib
import numpy as np


class ModelUnavailableError(RuntimeError):
    pass


class SentimentAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model = None

    def load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise ModelUnavailableError(
                    "Le modèle est absent. Exécutez le script de réentraînement."
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, tweets: list[str]) -> dict[str, float]:
        model = self.load()
        positive = model["positive"].predict_proba(tweets)[:, 1]
        negative = model["negative"].predict_proba(tweets)[:, 1]
        scores = np.clip(positive - negative, -1.0, 1.0)
        return {
            tweet: round(float(score), 4)
            for tweet, score in zip(tweets, scores, strict=True)
        }

