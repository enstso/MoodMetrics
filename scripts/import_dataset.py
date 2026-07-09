import argparse

from moodmetrics.config import Config
from moodmetrics.datasets import import_tweet_rows


def main():
    parser = argparse.ArgumentParser(description="Importe un dataset CSV de tweets annotés.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remplace les tweets existants au lieu d'ajouter uniquement les nouveaux.",
    )
    args = parser.parse_args()

    imported_count = import_tweet_rows(
        Config.DATABASE_URL,
        Config.DATASET_PATH,
        replace=args.replace,
    )
    print(f"{imported_count} tweets importés depuis {Config.DATASET_PATH}.")


if __name__ == "__main__":
    main()
