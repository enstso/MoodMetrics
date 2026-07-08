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

