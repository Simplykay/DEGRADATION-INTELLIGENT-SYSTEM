
from pathlib import Path
import json
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

CT_THRESHOLD = 60
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="Cotton Seed Degradation Intelligence API", version="1.0.0")

clf = joblib.load(MODELS_DIR / "degradation_classifier.joblib")
reg = joblib.load(MODELS_DIR / "ct_regressor.joblib")
rate_path = MODELS_DIR / "degradation_rate_model.joblib"
rate_model = joblib.load(rate_path) if rate_path.exists() else None
scored = pd.read_csv(ARTIFACT_DIR / "stakeholder_scored_batches.csv")

class PredictRequest(BaseModel):
    batch_CT_initial: float
    weak_fraction: float
    stage: float
    days_since_initial: float
    days_since_lot_created: float
    max_cumulated_dd60: float
    avg_soil_moisture: float
    total_net_solar_radiation: float
    season_length: float
    rm: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows_scored": int(len(scored)),
        "ct_threshold": CT_THRESHOLD
    }

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([req.model_dump()])
    prob = clf.predict_proba(df)[0, 1] if hasattr(clf, "predict_proba") else float(clf.predict(df)[0])
    pred_ct = float(reg.predict(df)[0])
    pred_rate = max(float(rate_model.predict(df)[0]), 0.01) if rate_model is not None else 0.05
    days_to_threshold = max((pred_ct - CT_THRESHOLD) / pred_rate, 0)
    return {
        "prob_degradation": round(float(prob), 4),
        "pred_ct_current": round(pred_ct, 2),
        "days_to_threshold_proxy": round(days_to_threshold, 2),
        "predicted_risk_bucket": "At Risk" if pred_ct <= 60 else ("Watch" if pred_ct <= 70 else "Safe")
    }

@app.get("/top_lots")
def top_lots(limit: int = 25):
    cols = [c for c in [
        "stakeholder_rank", "batch_id", "lineage_id", "stage", "pred_prob_degradation",
        "pred_ct_current", "days_to_threshold_proxy", "predicted_risk_bucket"
    ] if c in scored.columns]
    out = scored.sort_values(["stakeholder_rank", "pred_prob_degradation"], ascending=[True, False])[cols].head(limit)
    return out.to_dict(orient="records")

@app.post("/what_if")
def what_if(req: PredictRequest):
    return predict(req)
