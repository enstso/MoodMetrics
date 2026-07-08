import os


class Config:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://moodmetrics:moodmetrics@localhost:3306/moodmetrics",
    )
    MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/sentiment_model.joblib")

