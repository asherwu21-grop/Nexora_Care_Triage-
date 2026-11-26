"""Documentation must stay synchronized with artifact values and case-study criteria."""

import re
from copy import deepcopy
from urllib.parse import unquote

import pytest

from src.config import ARTIFACTS_DIR, PROJECT_ROOT
from src.reporting import Artifacts, build_context, generate, percent


def test_generated_documents_and_exact_criteria_are_current():
    outputs = generate(check=True)
    reference = Artifacts().read("case_study_reference.json")
    for name in ["README.md", "docs/project_report.md"]:
        text = outputs[name]
        assert "${" not in text
        for criterion in reference["success_criteria"]:
            assert text.count(criterion) == 1
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith("https:"):
                continue
            path = ((PROJECT_ROOT / name).parent / unquote(target)).resolve()
            if path.suffix == ".joblib":  # Explicitly documented separate local distribution.
                continue
            assert path.exists(), f"Broken documentation link: {target} in {name}"


def test_documented_accuracy_tracks_changed_artifact_instead_of_literal():
    class ChangedMetric(Artifacts):
        def read(self, name):
            value = super().read(name)
            if name == "model_metrics.json":
                value = deepcopy(value)
                value["accuracy"] = 0.123456789
            return value

    original = build_context(Artifacts())
    changed = build_context(ChangedMetric())
    assert changed["accuracy"] == percent(0.123456789)
    assert original["accuracy"] != changed["accuracy"]
    assert original["macro_f1"] == changed["macro_f1"]


def test_inconsistent_test_support_is_rejected():
    class InconsistentSupport(Artifacts):
        def read(self, name):
            value = super().read(name)
            if name == "model_metrics.json":
                value["n_evaluated"] += 1
            return value

    with pytest.raises(ValueError, match="Confusion matrix support"):
        build_context(InconsistentSupport())


def test_templates_contain_no_literal_numeric_measurements():
    for path in (PROJECT_ROOT / "docs/templates").glob("*.tmpl"):
        text = path.read_text()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`]*`", "", text)
        text = re.sub(r"\]\([^)]*\)", "]", text)
        # F1/TF-IDF and paths are identifiers; standalone numeric claims must be placeholders.
        assert not re.search(r"(?<![\w])\d+(?:\.\d+)?(?![\w])", text), path.name
    reader = Artifacts()
    build_context(reader)
    assert reader.root == ARTIFACTS_DIR
    assert "model_metrics.json" in reader.sources
    assert "case_study_reference.json" in reader.sources
