from moodmetrics.modeling import (
    BASELINE_MODEL_NAME,
    HYBRID_CHAR_WORD_MODEL_NAME,
    OPTIMAL_MODEL_NAME,
    get_model_spec,
    iter_model_specs,
)


def test_get_model_spec_returns_optimal_by_default():
    spec = get_model_spec()

    assert spec.name == OPTIMAL_MODEL_NAME == HYBRID_CHAR_WORD_MODEL_NAME
    assert spec.to_metadata()["attempt"] == "tentative_2_hybrid_features"


def test_baseline_model_spec_is_still_available():
    spec = get_model_spec(BASELINE_MODEL_NAME)

    assert spec.to_metadata()["attempt"] == "baseline"


def test_model_specs_build_predict_proba_pipeline():
    texts = ["service excellent", "service horrible", "livraison reçue", "super app"]
    labels = [True, False, False, True]

    for spec in iter_model_specs():
        model = spec.build().fit(texts, labels)

        assert hasattr(model, "predict_proba")
        assert model.predict(["excellent"]).tolist() == [True]


def test_hybrid_model_spec_is_registered():
    spec = get_model_spec(HYBRID_CHAR_WORD_MODEL_NAME)

    assert spec.attempt == "tentative_2_hybrid_features"
