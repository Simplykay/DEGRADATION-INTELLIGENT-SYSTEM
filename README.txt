Cotton Seed Degradation Intelligence — Stakeholder Bundle

Contents:
- stakeholder_upgrade_single_notebook.ipynb
- app_bundle/streamlit/streamlit_app.py
- app_bundle/api/fastapi_app.py
- saved model artifacts in stakeholder_outputs/models
- stakeholder CSV artifacts in stakeholder_outputs/artifacts

Suggested run order:
1. Run the notebook top to bottom.
2. Confirm that model artifacts and CSV outputs were written.
3. Streamlit:
   streamlit run stakeholder_outputs/app_bundle/streamlit/streamlit_app.py
4. FastAPI:
   uvicorn stakeholder_outputs.app_bundle.api.fastapi_app:app --reload

Stakeholder-ready features:
- degradation probability
- predicted CT
- days-to-threshold proxy
- priority queue
- SHAP driver tab
- what-if simulation