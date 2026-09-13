"""The Streamlit demo the `ui` dependency group declared but never shipped.

Three tabs, one per module: the point-in-time feature store (query a value as of a
moment in the past, watch it change once a later write becomes "knowable"), skew
detection (compare a training batch against a serving batch), and the release gate
(fill in a model card + metrics + fairness + skew, see the gate's verdict with the
specific failures named).

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from insurance.featurestore import FeatureStore  # noqa: E402
from insurance.governance import ModelCard, ReleaseGate  # noqa: E402
from insurance.skew import detect_skew  # noqa: E402

st.set_page_config(page_title="insurance-mlops demo", layout="wide")
st.title("insurance-mlops")
st.caption(
    "Point-in-time feature store, training/serving skew, and a release gate that "
    "refuses — the three modules this repo's tests exercise, made visible."
)

tab_store, tab_skew, tab_gate = st.tabs(
    ["Point-in-time feature store", "Training/serving skew", "Release gate"]
)

# ---- Tab 1: feature store -----------------------------------------------------

with tab_store:
    st.markdown(
        "A claim is **filed** on day 0, but the system only learns about it (and can "
        "only ever use it as a feature) some days later, on `available_at`. Querying "
        "`as_of` a moment *before* `available_at` must not see the value — that is the "
        "whole point of a point-in-time store."
    )

    if "store" not in st.session_state:
        st.session_state.store = FeatureStore()
        st.session_state.store.write("cust-1", "prior_claims", 1, event_time=0)
        st.session_state.store.write("cust-1", "prior_claims", 4, event_time=60, available_at=65)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**History written so far** (entity `cust-1`, feature `prior_claims`)")
        series = st.session_state.store.history_for("cust-1", "prior_claims")
        st.table(
            [
                {"value": r.value, "event_time": r.event_time, "available_at": r.available_at}
                for r in series
            ]
        )
        new_value = st.number_input("Add a new value", value=10, key="new_value")
        new_event_time = st.number_input("event_time", value=90, key="new_event_time")
        new_available_at = st.number_input(
            "available_at (when the system could first know it)",
            value=95,
            key="new_available_at",
        )
        if st.button("Write"):
            st.session_state.store.write(
                "cust-1", "prior_claims", new_value, new_event_time, new_available_at
            )
            st.rerun()

    with col2:
        as_of = st.slider("Query as_of", min_value=0, max_value=120, value=30)
        value = st.session_state.store.get_as_of("cust-1", "prior_claims", as_of)
        staleness = st.session_state.store.staleness("cust-1", "prior_claims", as_of)
        st.metric("Value visible at this moment", value.value if value else "None")
        if staleness is not None:
            st.metric("Staleness (seconds)", f"{staleness:.0f}")
        st.info(
            "Move the slider past `available_at = 65` and the value jumps from 1 to 4 — "
            "not at `event_time = 60`, because the model couldn't have known yet."
        )

# ---- Tab 2: skew ---------------------------------------------------------------

with tab_skew:
    st.markdown(
        "Skew is two pipelines disagreeing, not the world changing. Generate a "
        "**clean** pair (should find nothing) or a **skewed** pair (training computed "
        "age in years, serving computed it in months) and compare."
    )

    scenario = st.radio("Scenario", ["Clean (no skew)", "Unit mismatch (years vs months)"])

    training = [{"age": v} for v in [25, 30, 35, 40, 45, 50] * 10]
    if scenario == "Clean (no skew)":
        serving = [{"age": v} for v in [26, 31, 34, 41, 44, 51] * 10]
    else:
        serving = [{"age": v * 12} for v in [25, 30, 35, 40, 45, 50] * 10]

    if st.button("Run skew detection"):
        report = detect_skew(training, serving)
        if report["safe_to_serve"]:
            st.success(f"safe_to_serve = True — {report['warnings']} warning(s), 0 critical")
        else:
            st.error(f"safe_to_serve = False — {report['critical']} critical finding(s)")
        for finding in report["findings"]:
            severity_icon = "🔴" if finding["severity"] == "critical" else "🟡"
            st.write(f"{severity_icon} **{finding['feature']}** ({finding['kind']}): {finding['detail']}")

# ---- Tab 3: release gate ---------------------------------------------------------

with tab_gate:
    st.markdown("Fill in a model card and metrics, then see whether the release gate approves.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model card")
        owner = st.text_input("Owner", value="claims-ml-team")
        reviewed_by = st.text_input("Reviewed by", value="")
        intended_use = st.text_input("Intended use", value="Claims triage priority scoring")
        out_of_scope = st.text_input("Out of scope (comma-separated)", value="pricing, underwriting")
        training_data = st.text_input("Training data", value="2024-2025 claims, point-in-time joined")
        limitations = st.text_input("Limitations (comma-separated)", value="Not validated outside Punjab")

    with col2:
        st.subheader("Metrics & fairness")
        gini = st.slider("Gini", 0.0, 1.0, 0.42)
        brier = st.slider("Brier score", 0.0, 1.0, 0.18)
        disparate_impact = st.slider("Worst disparate impact ratio", 0.0, 1.0, 0.88)
        has_skew = st.checkbox("Skew detector reported critical findings", value=False)

    if st.button("Evaluate release gate", type="primary"):
        card = ModelCard(
            name="claims-triage",
            version="1.0",
            owner=owner,
            reviewed_by=reviewed_by,
            intended_use=intended_use,
            out_of_scope=[s.strip() for s in out_of_scope.split(",") if s.strip()],
            training_data=training_data,
            limitations=[s.strip() for s in limitations.split(",") if s.strip()],
        )
        gate = ReleaseGate()
        result = gate.evaluate(
            card=card,
            metrics={"gini": gini, "brier": brier},
            fairness={"worst_disparate_impact": disparate_impact},
            skew={"safe_to_serve": not has_skew, "critical": 1 if has_skew else 0},
        )
        if result["approved"]:
            st.success(f"APPROVED — gate `{result['gate']}`")
        else:
            st.error(f"REFUSED — gate `{result['gate']}`")
            for failure in result["failures"]:
                st.write(f"- {failure}")
