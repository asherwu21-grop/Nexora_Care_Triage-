"""AppTest smoke coverage plus interactive safety and input-validation regressions."""

from datetime import date, timedelta
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.utils import data_loader as dl
from src.config import PROJECT_ROOT
from src.dashboard_logic import build_demo_row, error_table
from src.dashboard_views import PAGE_TITLES
from src.inference import PRIOR_COUNT, SAFETY_NOTICE

APP = PROJECT_ROOT / "dashboard/app.py"
PAGES = [
    ("overview", None),
    ("eda", "01_Exploratory_Analysis.py"),
    ("hypotheses", "02_Hypothesis_Testing.py"),
    ("evaluation", "03_Model_Evaluation.py"),
    ("errors", "04_Error_Analysis.py"),
    ("explainability", "05_Explainability.py"),
    ("prediction", "06_Prediction_Demo.py"),
]


def open_page(key):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    page = dict(PAGES)[key]
    if page:
        app.switch_page(f"pages/{page}").run()
    assert not app.exception
    return app


@pytest.mark.parametrize("key,page", PAGES)
def test_each_dashboard_page_renders(key, page):
    app = open_page(key)
    assert app.title[0].value == PAGE_TITLES[key]
    assert any(SAFETY_NOTICE in item.value for item in app.warning)
    assert not any("unavailable" in item.value for item in app.error)
    if key != "prediction":
        assert app.get("plotly_chart") or app.get("imgs")


def test_eda_filters_empty_and_all_exploration_views():
    app = open_page("eda")
    app.multiselect(key="filter_urgency").set_value([]).run()
    assert not app.exception
    assert any("No requests match" in item.value for item in app.info)
    app.multiselect(key="filter_urgency").set_value(["Critical"]).run()
    for view in [
        "Symptoms",
        "Red flags",
        "Comorbidities",
        "Vitals and missingness",
        "Numeric correlations",
    ]:
        app.selectbox[0].select(view).run()
        assert not app.exception
        assert not app.error
        assert app.get("plotly_chart")


def test_hypothesis_evidence_and_patient_sensitivity_controls():
    app = open_page("hypotheses")
    app.radio[0].set_value("Not significant").run()
    assert not app.exception
    table = app.dataframe[0].value
    assert table["Evidence after Holm"].eq("Not significant").all()
    app.selectbox[0].select_index(1).run()
    assert not app.exception


def complete_form(app):
    app.date_input(key="demo_dob").set_value(date(1984, 1, 1))
    app.checkbox(key="demo_history").check()
    return app


def submit(app):
    app.button[0].click().run()
    assert not app.exception
    return app


def test_demo_valid_raw_input_probabilities_and_red_flag_override(monkeypatch):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.7, 0.1, 0.1, 0.1]])
    monkeypatch.setattr(dl, "load_model", lambda: model)
    app = open_page("prediction")
    app.multiselect(key="demo_flags").set_value(["Chest pain"]).run()
    assert any("ESCALATION NOTICE" in item.value for item in app.error)
    model.predict_proba.assert_not_called()
    submit(complete_form(app))
    assert any("Low" in item.value for item in app.subheader)
    assert any("ESCALATION NOTICE" in item.value for item in app.error)
    assert any(SAFETY_NOTICE in item.value for item in app.warning)
    np.testing.assert_allclose(app.dataframe[0].value.Probability.sum(), 1)
    model.predict_proba.assert_called_once()
    raw = model.predict_proba.call_args.args[0]
    assert raw.loc[0, "red_flags"] == "Chest pain"
    assert raw.loc[0, "request_id"].startswith("demo-request-")
    assert raw.loc[0, "heart_rate"] != raw.loc[0, "heart_rate"]  # Missing stays missing.
    assert "clinical_urgency" not in raw


def test_invalid_demo_does_not_call_model(monkeypatch):
    load = Mock(side_effect=AssertionError("Invalid form must never load the model"))
    monkeypatch.setattr(dl, "load_model", load)
    app = open_page("prediction")
    app.text_input(key="demo_heart_rate").set_value("not a number")
    submit(app)
    assert any("finite number" in item.value for item in app.error)
    assert any("Date of birth" in item.value for item in app.error)
    assert any("Confirm" in item.value for item in app.error)
    load.assert_not_called()


def test_missing_model_preserves_escalation(monkeypatch):
    def unavailable():
        raise dl.DashboardDataError("Missing model")

    monkeypatch.setattr(dl, "load_model", unavailable)
    app = open_page("prediction")
    app.multiselect(key="demo_flags").set_value(["Confusion"]).run()
    submit(complete_form(app))
    assert any("ESCALATION NOTICE" in item.value for item in app.error)
    assert any("Prediction unavailable" in item.value for item in app.error)


def test_missing_artifact_shows_help_instead_of_crashing(monkeypatch):
    def unavailable(name):
        raise dl.DashboardDataError("Missing resource")

    monkeypatch.setattr(dl, "load_json", unavailable)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any("unavailable" in item.value for item in app.error)


def test_demo_actual_saved_pipeline():
    if not (PROJECT_ROOT / "models/triage_model.joblib").exists():
        pytest.skip("Local Git-ignored final model not available")
    app = open_page("prediction")
    submit(complete_form(app))
    assert not app.error
    table = app.dataframe[0].value
    assert len(table) == 4
    np.testing.assert_allclose(table.Probability.sum(), 1)


def test_demo_validation_and_sentinel_semantics():
    values = {
        "date_of_birth": date(1984, 1, 1),
        "history_confirmed": True,
        PRIOR_COUNT: 0,
        "sex": "Female",
        "pregnancy_flag": "No",
        "chief_complaint_category": "Other",
        "symptom_severity": "Mild",
        "symptom_onset": "Gradual",
        "pain_level": "No pain",
        "red_flags_status": "Known none",
        "comorbidities_status": "Unknown",
    }
    row, issues = build_demo_row(values)
    assert not issues
    assert row.loc[0, "red_flags"] == "No red flags"
    assert row.comorbidities.isna().all()
    row, issues = build_demo_row(
        {
            **values,
            "heart_rate": "999",
            "duration_hours": "-1",
            "date_of_birth": date.today() + timedelta(days=1),
        }
    )
    assert row is None and len(issues) == 3
    row, issues = build_demo_row({**values, "systolic_bp": "80", "diastolic_bp": "120"})
    assert row is None and any("Systolic" in issue for issue in issues)


def test_error_review_broad_focus_is_separate_from_severe():
    import pandas as pd

    predictions = pd.DataFrame(
        {"request_id": ["a", "b", "c"], "predicted_urgency": ["High", "Moderate", "Low"]}
    )
    dataset = pd.DataFrame(
        {
            "request_id": ["a", "b", "c"],
            "clinical_urgency": ["Critical", "High", "High"],
            "patient_age": [20] * 3,
            "sex": ["Female"] * 3,
            "chief_complaint_category": ["Other"] * 3,
        }
    )
    result = error_table(predictions, dataset)
    assert result.safety_focus.tolist() == [True, True, True]
    assert result.severe_under_triage.tolist() == [False, False, True]


def test_loader_rejects_path_escape_and_invalidates_changed_data(tmp_path, monkeypatch):
    with pytest.raises(dl.DashboardDataError):
        dl.load_table("../../AGENTS.md")
    monkeypatch.setattr(dl, "ARTIFACTS_DIR", tmp_path)
    path = tmp_path / "counts.csv"
    path.write_text("count\n1\n")
    assert dl.load_table("counts.csv").iloc[0, 0] == 1
    path.write_text("count\n200\n")
    assert dl.load_table("counts.csv").iloc[0, 0] == 200
