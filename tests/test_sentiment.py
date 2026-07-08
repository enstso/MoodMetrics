import numpy as np

from moodmetrics.sentiment import SentimentAnalyzer


class FakeClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, tweets):
        return np.array([[1 - value, value] for value in self.probabilities])


def test_score_is_positive_probability_minus_negative_probability():
    analyzer = SentimentAnalyzer("unused.joblib")
    analyzer._model = {
        "positive": FakeClassifier([0.9, 0.2]),
        "negative": FakeClassifier([0.1, 0.8]),
    }

    assert analyzer.predict(["bon", "mauvais"]) == {"bon": 0.8, "mauvais": -0.6}

