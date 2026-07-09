import os


class Config:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://moodmetrics:moodmetrics@localhost:3306/moodmetrics",
    )
    DATASET_PATH = os.getenv("DATASET_PATH", "data/tweets.csv")
    EVALUATION_PATH = os.getenv("EVALUATION_PATH", "reports/evaluation.json")
    MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/sentiment_model.joblib")
    REPORT_PATH = os.getenv("REPORT_PATH", "reports/evaluation_report.pdf")
