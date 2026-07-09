import pytest

from moodmetrics import create_app


class FakeAnalyzer:
    def predict(self, tweets):
        return {tweet: 0.5 for tweet in tweets}


@pytest.fixture
def client():
    app = create_app({"TESTING": True}, analyzer=FakeAnalyzer())
    return app.test_client()


def test_analyze_tweets(client):
    response = client.post(
        "/api/v1/sentiments",
        json={"tweets": ["Excellent service", "Mauvaise expérience"]},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "Excellent service": 0.5,
        "Mauvaise expérience": 0.5,
    }


@pytest.mark.parametrize("payload", [None, {}, {"tweets": []}, {"tweets": [""]}])
def test_rejects_invalid_payload(client, payload):
    response = client.post("/api/v1/sentiments", json=payload)

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_accepts_raw_array(client):
    response = client.post("/api/v1/sentiments", json=["Très bien"])

    assert response.status_code == 200
    assert response.get_json() == {"Très bien": 0.5}


def test_model_metrics_returns_evaluation_payload(tmp_path):
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        '{"samples": 12, "labels": {"positive": {"accuracy": 0.9}}}',
        encoding="utf-8",
    )
    app = create_app(
        {"TESTING": True, "EVALUATION_PATH": str(evaluation_path)},
        analyzer=FakeAnalyzer(),
    )

    response = app.test_client().get("/api/v1/model/metrics")

    assert response.status_code == 200
    assert response.get_json() == {
        "samples": 12,
        "labels": {"positive": {"accuracy": 0.9}},
    }


def test_model_metrics_returns_503_when_missing(tmp_path):
    app = create_app(
        {"TESTING": True, "EVALUATION_PATH": str(tmp_path / "missing.json")},
        analyzer=FakeAnalyzer(),
    )

    response = app.test_client().get("/api/v1/model/metrics")

    assert response.status_code == 503
    assert "error" in response.get_json()
