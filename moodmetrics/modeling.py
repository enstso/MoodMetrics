from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    attempt: str
    build: Callable[[], Pipeline]

    def to_metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "attempt": self.attempt,
        }


BASELINE_MODEL_NAME = "baseline_word_tfidf_logreg"
OPTIMAL_MODEL_NAME = BASELINE_MODEL_NAME


def build_baseline_word_tfidf_logreg() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1_000, class_weight="balanced"),
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    BASELINE_MODEL_NAME: ModelSpec(
        name=BASELINE_MODEL_NAME,
        description="Baseline TF-IDF mots 1-2 grammes + LogisticRegression.",
        attempt="baseline",
        build=build_baseline_word_tfidf_logreg,
    )
}


def get_model_spec(name: str | None = None) -> ModelSpec:
    model_name = name or OPTIMAL_MODEL_NAME
    try:
        return MODEL_SPECS[model_name]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Modèle inconnu: {model_name}. Disponibles: {available}.") from error


def iter_model_specs(names: Iterable[str] | None = None) -> list[ModelSpec]:
    if names is None:
        return list(MODEL_SPECS.values())
    return [get_model_spec(name) for name in names]
