from moodmetrics.database import Base, Tweet, create_session_factory
from scripts.optimize_model import optimize_from_database


def test_optimize_from_database_ranks_candidates(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tweets.db'}"
    output_path = tmp_path / "optimization.json"
    session_factory, engine = create_session_factory(database_url)
    Base.metadata.create_all(engine)

    with session_factory() as session:
        session.add_all(
            [
                Tweet(text=f"super rapide excellent {index}", positive=True, negative=False)
                for index in range(10)
            ]
        )
        session.add_all(
            [
                Tweet(text=f"lent bug horrible {index}", positive=False, negative=True)
                for index in range(10)
            ]
        )
        session.add_all(
            [
                Tweet(text=f"information compte livraison {index}", positive=False, negative=False)
                for index in range(10)
            ]
        )
        session.commit()

    optimization = optimize_from_database(database_url, str(output_path), test_size=0.3)

    assert output_path.exists()
    assert optimization["samples"] == 30
    assert len(optimization["attempts"]) >= 3
    assert "f1_score" in optimization["attempts"][0]["aggregate"]
