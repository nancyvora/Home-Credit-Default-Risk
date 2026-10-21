"""The scoring service: a FastAPI app with a Gradio form mounted at ``/ui``.

Endpoints:

- ``GET  /``          health check, with the model's name and number of features
- ``GET  /model``     what the model is, how it was measured (cross-validation, holdout, Kaggle), and its thresholds
- ``POST /predict``   one applicant's raw records -> probability of payment difficulties, decision, main reasons
- ``/ui``             a form for trying the model on the application fields that matter most

Every score goes through ``src/serving/inference.py``, the same code as the batch submission, which builds the
features with the training preprocessing and feature code.

Run with ``python -m src.app.main`` (or ``uvicorn src.app.app:app``) and open http://127.0.0.1:8000/docs or /ui.
"""
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.serving.inference import HISTORY_TABLES, MODEL_DIR, load_model, predict

app = FastAPI(
    title='Home Credit default risk',
    description='Probability that a loan applicant will have payment difficulties, from the application and their credit history.',
    version='1.0.0',
)


class Application(BaseModel):
    """One row of application_test.csv. Only SK_ID_CURR is required; a column left out is treated as missing."""
    model_config = ConfigDict(extra='allow', json_schema_extra={'example': {
        'SK_ID_CURR': 100001, 'NAME_CONTRACT_TYPE': 'Cash loans', 'CODE_GENDER': 'F', 'AMT_INCOME_TOTAL': 135000.0,
        'AMT_CREDIT': 568800.0, 'AMT_ANNUITY': 20560.5, 'AMT_GOODS_PRICE': 450000.0, 'DAYS_BIRTH': -19241, 'DAYS_EMPLOYED': -2329,
        'NAME_EDUCATION_TYPE': 'Higher education', 'EXT_SOURCE_1': 0.75, 'EXT_SOURCE_2': 0.79, 'EXT_SOURCE_3': 0.16}})
    SK_ID_CURR: int


class PredictRequest(BaseModel):
    application: Application
    history: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description=(
        f'Earlier records of the applicant, by table: {", ".join(HISTORY_TABLES)}. Each record is a row of the Kaggle file of that '
        'table. A table left out means the applicant has no records in it.'))
    threshold: Optional[float] = Field(None, gt=0, lt=1, description='decline at or above this probability; default: the reference threshold')


@app.get('/')
def health():
    model = load_model()
    return {'status': 'ok', 'model': model.metadata['name'], 'features': len(model.features)}


@app.get('/model')
def model_card():
    model = load_model()
    meta = model.metadata
    evaluation_path = MODEL_DIR / 'holdout_evaluation.json'
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8')) if evaluation_path.exists() else {}
    return {'name': meta['name'], 'predicts': meta['predicts'], 'features': len(model.features),
            'trained_on': {'rows': meta['training']['rows'], 'defaults': meta['training']['defaults']},
            'decision': meta['decision'], 'evaluation': {k: evaluation.get(k) for k in ('cross_validation', 'holdout', 'kaggle')},
            'known_limits': meta.get('known_limits', [])}


@app.post('/predict')
def api_predict(request: PredictRequest):
    try:
        return predict(request.application.model_dump(), request.history, threshold=request.threshold)
    except ValueError as e:                     # bad input: unknown table, records of another applicant
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------- Gradio interface (src/app/ui.py)

from src.app.ui import build_ui, score_form, score_json          # noqa: E402  (score_form is used by the tests)

try:
    import gradio as gr
    app = gr.mount_gradio_app(app, build_ui(), path='/ui')
except ImportError:                             # the API works without the interface
    pass
