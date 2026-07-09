import json
from pathlib import Path

from flask import Flask, current_app, jsonify, request

from moodmetrics.config import Config
from moodmetrics.sentiment import ModelUnavailableError, SentimentAnalyzer


def create_app(config: dict | None = None, analyzer=None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config:
        app.config.update(config)

    app.extensions["sentiment_analyzer"] = analyzer or SentimentAnalyzer(
        app.config["MODEL_PATH"]
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/sentiments")
    def analyze_sentiments():
        payload = request.get_json(silent=True)
        tweets = payload.get("tweets") if isinstance(payload, dict) else payload

        if not isinstance(tweets, list):
            return jsonify(error="Le corps doit contenir une liste de tweets."), 400
        if not tweets:
            return jsonify(error="La liste de tweets ne peut pas être vide."), 400
        if any(not isinstance(tweet, str) or not tweet.strip() for tweet in tweets):
            return jsonify(error="Chaque tweet doit être une chaîne non vide."), 400

        try:
            result = current_app.extensions["sentiment_analyzer"].predict(tweets)
        except ModelUnavailableError as error:
            return jsonify(error=str(error)), 503

        return jsonify(result)

    @app.get("/api/v1/model/metrics")
    def model_metrics():
        evaluation_path = Path(current_app.config["EVALUATION_PATH"])
        if not evaluation_path.exists():
            return jsonify(error="Les métriques du modèle ne sont pas encore générées."), 503

        try:
            metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return jsonify(error="Le fichier de métriques du modèle est invalide."), 500

        return jsonify(metrics)

    return app


app = create_app()
