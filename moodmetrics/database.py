from sqlalchemy import Boolean, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Tweet(Base):
    __tablename__ = "tweets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    positive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    negative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


def create_session_factory(database_url: str):
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(bind=engine), engine

