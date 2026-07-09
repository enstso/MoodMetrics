from moodmetrics.modeling import BASELINE_MODEL_NAME, get_model_spec, iter_model_specs


def test_get_model_spec_returns_baseline_by_default():
    spec = get_model_spec()

    assert spec.name == BASELINE_MODEL_NAME
    assert spec.to_metadata()["attempt"] == "baseline"


def test_model_specs_build_predict_proba_pipeline():
    texts = ["service excellent", "service horrible", "livraison reçue", "super app"]
    labels = [True, False, False, True]

    for spec in iter_model_specs():
        model = spec.build().fit(texts, labels)

        assert hasattr(model, "predict_proba")
        assert model.predict(["excellent"]).tolist() == [True]
