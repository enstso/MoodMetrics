import json

from moodmetrics.database import Base, Tweet, create_session_factory
from scripts.evaluate_model import evaluate_from_database


def test_evaluate_from_database_writes_metrics(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tweets.db'}"
    output_path = tmp_path / "evaluation.json"
    session_factory, engine = create_session_factory(database_url)
    Base.metadata.create_all(engine)

    with session_factory() as session:
        session.add_all(
            [
                Tweet(text=f"super service rapide {index}", positive=True, negative=False)
                for index in range(12)
            ]
        )
        session.add_all(
            [
                Tweet(text=f"mauvais bug lent {index}", positive=False, negative=True)
                for index in range(12)
            ]
        )
        session.add_all(
            [
                Tweet(text=f"message information neutre {index}", positive=False, negative=False)
                for index in range(12)
            ]
        )
        session.commit()

    evaluation = evaluate_from_database(database_url, str(output_path), test_size=0.3)
    saved_evaluation = json.loads(output_path.read_text(encoding="utf-8"))

    assert evaluation["samples"] == 36
    assert saved_evaluation["labels"]["positive"]["test_samples"] == 11
    assert saved_evaluation["labels"]["negative"]["confusion_matrix"]["labels"] == [
        "absent",
        "present",
    ]
