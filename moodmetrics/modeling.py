from collections.abc import Callable, Iterable
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline


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
TUNED_WORD_MODEL_NAME = "tuned_word_tfidf_logreg"
HYBRID_CHAR_WORD_MODEL_NAME = "hybrid_char_word_tfidf_logreg"
OPTIMAL_MODEL_NAME = HYBRID_CHAR_WORD_MODEL_NAME


def build_baseline_word_tfidf_logreg() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1_000, class_weight="balanced"),
    )


def build_tuned_word_tfidf_logreg() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            strip_accents="unicode",
            sublinear_tf=True,
            max_features=30_000,
        ),
        LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            C=1.5,
            solver="liblinear",
        ),
    )


def build_hybrid_char_word_tfidf_logreg() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    max_features=25_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    max_features=20_000,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_500,
                    class_weight="balanced",
                    C=2.0,
                    solver="liblinear",
                ),
            ),
        ]
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    BASELINE_MODEL_NAME: ModelSpec(
        name=BASELINE_MODEL_NAME,
        description="Baseline TF-IDF mots 1-2 grammes + LogisticRegression.",
        attempt="baseline",
        build=build_baseline_word_tfidf_logreg,
    ),
    TUNED_WORD_MODEL_NAME: ModelSpec(
        name=TUNED_WORD_MODEL_NAME,
        description=(
            "Tentative 1: TF-IDF mots 1-3 grammes, accents normalisés, "
            "pondération sublinear_tf et C=1.5."
        ),
        attempt="tentative_1_tfidf",
        build=build_tuned_word_tfidf_logreg,
    ),
    HYBRID_CHAR_WORD_MODEL_NAME: ModelSpec(
        name=HYBRID_CHAR_WORD_MODEL_NAME,
        description=(
            "Tentative 2: union TF-IDF mots + char_wb 3-5 grammes pour mieux "
            "capturer fautes, hashtags, emojis et variations courtes."
        ),
        attempt="tentative_2_hybrid_features",
        build=build_hybrid_char_word_tfidf_logreg,
    ),
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
