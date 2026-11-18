"""
early_warning_page.py — Part 9: the "Early Warning & Recommendation"
Streamlit page. Read-only consumer of the frozen decision_support module;
imports no training code path.

Run: streamlit run early_warning_page.py
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_support as ds

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inputs"))
from phase6_data_prep import load_frozen_dataset, FROZEN_FEATURES

st.set_page_config(page_title="Locust Early Warning", layout="centered")

RISK_COLORS = {"LOW": "#2e7d32", "MODERATE": "#e6a700", "HIGH": "#c62828"}


@st.cache_data
def load_test_rows():
    _cwd = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inputs"))
    df = load_frozen_dataset()
    os.chdir(_cwd)
    return df[df["split"] == "test"].reset_index(drop=True)


RISK_BG = {"LOW": "#eaf5ea", "MODERATE": "#fbf2df", "HIGH": "#fbe9e7"}

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div style="padding:4px 0 2px 0;">
      <span style="background:#eef1f7; color:#33415c; font-size:12px; font-weight:600;
                   padding:4px 10px; border-radius:12px; letter-spacing:0.02em;">
        RESEARCH SIMULATION -- NOT A LIVE DEPLOYMENT
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Desert Locust Early Warning System")
st.caption(
    "Federated Learning-Based Early Warning and Recommendation System for Desert Locust "
    "Onset Prediction -- decision-support layer, reading the frozen, multi-seed-validated "
    "Adaptive Aggregation (Method A) model. Predicts observed-report onset -- not "
    "confirmed biological outbreak."
)

with st.expander("Who is this for?", expanded=False):
    st.markdown(
        "**Primary users:** agricultural authorities, locust-monitoring authorities, "
        "agricultural extension officers, and field surveillance teams -- the system flags "
        "regions that warrant increased monitoring.\n\n"
        "**Farmers are indirect beneficiaries**, who would receive any resulting warning "
        "through extension services or other local communication channels, not through this "
        "dashboard directly. A farmer-facing mobile, local-language, or voice interface is "
        "**future work**, not part of the current system."
    )

st.divider()

test_df = load_test_rows()
countries = sorted(test_df["COUNTRYNAME"].unique())
country = st.selectbox("Country", countries, index=countries.index("Kenya") if "Kenya" in countries else 0)

country_rows = test_df[test_df["COUNTRYNAME"] == country]
regions = sorted(country_rows["REGION"].unique())
region = st.selectbox("Region", regions)

region_rows = country_rows[country_rows["REGION"] == region]
periods = sorted(region_rows["period_T"].unique())
if len(periods) == 0:
    st.warning("No available forecast period for this country/region combination in the frozen test set.")
    st.stop()
period = st.selectbox(
    "Forecast base month (features observed at this month; prediction is for the following month)",
    periods,
)

row = region_rows[region_rows["period_T"] == period].iloc[0]
feature_row = {f: float(row[f]) for f in FROZEN_FEATURES}

score = ds.predict_score(feature_row)
tier = ds.risk_tier(score)

st.divider()

st.subheader("Prediction")
col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted onset score", f"{score:.2f}")
    st.caption("Score (0-1), not a calibrated probability -- see Limitations.")
with col2:
    st.markdown(
        f"<div style='background:{RISK_BG[tier]}; border-left:4px solid {RISK_COLORS[tier]}; "
        f"padding:10px 14px; border-radius:6px;'><b style='color:{RISK_COLORS[tier]};'>"
        f"Risk level: {tier}</b></div>",
        unsafe_allow_html=True,
    )
st.caption("Decision-support thresholds -- not biologically validated thresholds "
           "(LOW < 0.30, MODERATE 0.30-0.60, HIGH > 0.60).")

st.subheader("Early Warning")
st.info(ds.early_warning_message(tier))

st.subheader("Recommendation")
st.success(ds.recommendation(tier))
st.caption("Fixed, deterministic rule table -- never free-form model output.")

st.divider()

st.subheader("Why?")
with st.spinner("Computing SHAP explanation..."):
    explanation = ds.explain(feature_row, top_k=3)

st.markdown("**Environmental contributors**")
for item in explanation["environmental_contributors"]:
    arrow = "up" if "positive" in item["direction"] else "down"
    st.markdown(f"- {item['feature']} -- {arrow} -- {item['magnitude_label']} ({item['direction']})")
st.caption("Contributed to the model prediction -- not a claim of scientific or biological causation.")

if explanation["data_quality_indicators"]:
    st.markdown("**Data-quality indicators**")
    for item in explanation["data_quality_indicators"]:
        st.markdown(f"- [!] {item['indicator']} -- {item['magnitude_label']} to the model score")
    st.caption(explanation["data_quality_disclaimer"])

st.divider()

st.subheader("Data Evidence")
evidence = ds.data_evidence(country)
ev_col1, ev_col2 = st.columns(2)
with ev_col1:
    st.metric("Training samples", evidence.get("train_samples", "n/a"))
    st.metric("Validation samples", evidence.get("val_samples", "n/a"))
with ev_col2:
    st.metric("Positive training examples", evidence.get("train_positive_examples", "n/a"))
    st.metric("Positive validation examples", evidence.get("val_positive_examples", "n/a"))
if "zero_positive_notice" in evidence:
    st.warning(evidence["zero_positive_notice"])
elif "sparse_history_notice" in evidence:
    st.warning(evidence["sparse_history_notice"])

st.divider()

with st.expander("Methodology"):
    st.markdown(
        "```\n"
        "Raw GLAD42 dataset (79,528 rows, 42 countries)\n"
        "        v  cleaning + T->T+1 transformation\n"
        "Engineered dataset (56,787 rows, 36 columns)\n"
        "        v  9 frozen features selected\n"
        "Country-level clients (21)\n"
        "        v  local logistic regression\n"
        "Federated aggregation (FedAvg / Adaptive A / Adaptive B)\n"
        "        v\n"
        "Frozen Method A model\n"
        "        v\n"
        "Onset score -> Risk level -> Early warning -> SHAP -> Recommendation\n"
        "```\n"
        "Adaptive aggregation reweights each client's contribution using only "
        "training-set statistics (sample count, positive count) -- never test data. "
        "It improved specific heterogeneous, data-sparse clients (e.g. Kenya, Uganda) "
        "relative to standard FedAvg, but **did not outperform FedAvg on average** across "
        "all evaluated clients -- this is reported as a client-dependent effect, not a "
        "universal improvement."
    )

with st.expander("Limitations"):
    for item in ds.LIMITATIONS:
        st.markdown(f"- {item}")
