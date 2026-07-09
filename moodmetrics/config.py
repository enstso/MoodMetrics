import os


class Config:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://moodmetrics:moodmetrics@localhost:3306/moodmetrics",
    )
    DATASET_PATH = os.getenv("DATASET_PATH", "data/tweets.csv")
    # Jeu de tweets réalistes supplémentaires (ironie, négation, argot, fautes...)
    # ajouté au pool d'évaluation pour rendre les métriques crédibles.
    SUPPLEMENTARY_DATASET_PATH = os.getenv(
        "SUPPLEMENTARY_DATASET_PATH", "data/tweets_supplementary.csv"
    )
    EVALUATION_PATH = os.getenv("EVALUATION_PATH", "reports/evaluation.json")
    MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/sentiment_model.joblib")
    REPORT_PATH = os.getenv("REPORT_PATH", "reports/evaluation_report.pdf")
