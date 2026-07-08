from moodmetrics.config import Config
from moodmetrics.database import Base, create_session_factory


def main():
    _, engine = create_session_factory(Config.DATABASE_URL)
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    main()

