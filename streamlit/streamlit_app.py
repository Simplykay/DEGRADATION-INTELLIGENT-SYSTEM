from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

CT_THRESHOLD = 60
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Cotton Seed Degradation Intelligence", layout="wide")
st.title("Cotton Seed Degradation Intelligence")
st.caption("Stakeholder edition: risk, time-to-threshold, drill-down, and what-if simulation")

@st.cache_data
def load_data():
    scored = pd.read_csv(ARTIFACT_DIR / "stakeholder_scored_batches.csv")
    feat = pd.read_csv(ARTIFACT_DIR / "stakeholder_feature_importance.csv")
    shap_imp = pd.read_csv(ARTIFACT_DIR / "stakeholder_shap_importance.csv")
    with open(ARTIFACT_DIR / "stakeholder_model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return scored, feat, shap_imp, meta

def patch_imputers(obj, visited=None):
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return
    visited.add(id(obj))
    
    if type(obj).__name__ == "SimpleImputer" and not hasattr(obj, "_fill_dtype"):
        obj._fill_dtype = np.float64
        
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if isinstance(v, list) or isinstance(v, tuple):
                for item in v:
                    patch_imputers(item, visited)
            else:
                patch_imputers(v, visited)

@st.cache_resource
def load_models():
    clf = joblib.load(MODELS_DIR / "degradation_classifier.joblib")
    reg = joblib.load(MODELS_DIR / "ct_regressor.joblib")
    rate_path = MODELS_DIR / "degradation_rate_model.joblib"
    rate = joblib.load(rate_path) if rate_path.exists() else None
    
    # Patch older scikit-learn models
    patch_imputers(clf)
    patch_imputers(reg)
    if rate is not None:
        patch_imputers(rate)
        
    return clf, reg, rate

scored, feat, shap_imp, meta = load_data()
clf, reg, rate_model = load_models()

with st.sidebar:
    st.header("Filters")
    stages = sorted(scored["stage"].dropna().unique().tolist()) if "stage" in scored.columns else []
    selected_stages = st.multiselect("Stage", stages, default=stages)
    selected_risk = st.multiselect("Predicted risk", sorted(scored["predicted_risk_bucket"].dropna().unique().tolist()), default=sorted(scored["predicted_risk_bucket"].dropna().unique().tolist()))
    max_prob = float(scored["pred_prob_degradation"].max()) if not scored.empty else 1.0
    prob_cut = st.slider("Minimum degradation probability", 0.0, 1.0, 0.0, 0.01)

f = scored.copy()
if selected_stages:
    f = f[f["stage"].isin(selected_stages)]
if selected_risk:
    f = f[f["predicted_risk_bucket"].isin(selected_risk)]
f = f[f["pred_prob_degradation"] >= prob_cut]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lots", len(f))
c2.metric("% At Risk", round((f["predicted_risk_bucket"] == "At Risk").mean() * 100, 2) if len(f) else 0)
c3.metric("Avg Pred CT", round(f["pred_ct_current"].mean(), 2) if len(f) else 0)
c4.metric("Avg Days To Threshold", round(f["days_to_threshold_proxy"].mean(), 1) if len(f) else 0)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Priority Queue", "Batch Drill-down", "Drivers", "What-if Simulator"])

with tab1:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(f, x="pred_ct_current", color="predicted_risk_bucket", nbins=30, title="Predicted CT distribution")
        fig.add_vline(x=CT_THRESHOLD, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = px.scatter(
            f,
            x="days_to_threshold_proxy",
            y="pred_prob_degradation",
            color="predicted_risk_bucket",
            hover_data=["batch_id", "lineage_id", "stage"],
            title="Risk vs time-to-threshold"
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(f.head(100), use_container_width=True)

with tab2:
    q = f.sort_values(["stakeholder_rank", "pred_prob_degradation"], ascending=[True, False]).copy()
    st.subheader("Top lots to prioritize")
    st.dataframe(q[[
        "stakeholder_rank", "batch_id", "lineage_id", "stage", "pred_prob_degradation",
        "pred_ct_current", "days_to_threshold_proxy", "predicted_risk_bucket"
    ]].head(100), use_container_width=True)

with tab3:
    ids = sorted(f["batch_id"].dropna().astype(str).unique().tolist())
    if ids:
        sel = st.selectbox("Select batch", ids)
        d = f[f["batch_id"].astype(str) == sel].copy()
        st.dataframe(d, use_container_width=True)
        mini = d[[
            "batch_id", "lineage_id", "stage", "batch_CT_initial", "CT_current_target", "pred_ct_current",
            "pred_prob_degradation", "days_to_threshold_proxy", "predicted_risk_bucket"
        ]]
        st.dataframe(mini, use_container_width=True)

with tab4:
    left, right = st.columns(2)
    with left:
        st.subheader("Global feature importance")
        st.dataframe(feat, use_container_width=True)
        if not feat.empty:
            st.plotly_chart(px.bar(feat.head(15), x="feature", y="importance", title="Top global drivers"), use_container_width=True)
    with right:
        st.subheader("SHAP summary")
        st.write(meta.get("shap_status"))
        st.dataframe(shap_imp, use_container_width=True)
        if not shap_imp.empty:
            st.plotly_chart(px.bar(shap_imp.head(15), x="feature", y="mean_abs_shap", title="Top SHAP drivers"), use_container_width=True)

with tab5:
    st.subheader("What-if simulation")
    sim = {}
    sim["batch_CT_initial"] = st.number_input("CT initial", 0.0, 100.0, 80.0)
    sim["weak_fraction"] = st.number_input("Weak fraction", 0.0, 100.0, 20.0)
    sim["stage"] = st.selectbox("Stage", [1, 2, 4], index=2)
    sim["days_since_initial"] = st.number_input("Days since initial", 0.0, 400.0, 30.0)
    sim["days_since_lot_created"] = st.number_input("Days since lot created", 0.0, 400.0, 30.0)
    sim["max_cumulated_dd60"] = st.number_input("Max cumulated DD60", 0.0, 5000.0, 1200.0)
    sim["avg_soil_moisture"] = st.number_input("Avg soil moisture", 0.0, 100.0, 20.0)
    sim["total_net_solar_radiation"] = st.number_input("Net solar radiation", 0.0, 500.0, 100.0)
    sim["season_length"] = st.number_input("Season length", 0.0, 300.0, 150.0)
    sim["rm"] = st.number_input("RM", 0.0, 10.0, 2.5)

    sim_df = pd.DataFrame([sim])
    prob = clf.predict_proba(sim_df)[0, 1] if hasattr(clf, "predict_proba") else float(clf.predict(sim_df)[0])
    pred_ct = float(reg.predict(sim_df)[0])
    if rate_model is not None:
        pred_rate = max(float(rate_model.predict(sim_df)[0]), 0.01)
    else:
        pred_rate = 0.05
    days_to_threshold = max((pred_ct - CT_THRESHOLD) / pred_rate, 0)

    a, b, c = st.columns(3)
    a.metric("Predicted degradation probability", round(prob, 4))
    b.metric("Predicted CT", round(pred_ct, 2))
    c.metric("Days to threshold proxy", round(days_to_threshold, 1))
