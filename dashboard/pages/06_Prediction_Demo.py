"""Nexora Care prediction view."""

import sys
from pathlib import Path

# Streamlit starts scripts with dashboard/ on sys.path; support the documented root command.
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.dashboard_views import render_page  # noqa: E402

render_page("prediction")
