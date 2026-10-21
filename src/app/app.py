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
from functools import lru_cache
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


# ---------------------------------------------------------------- Gradio form

EDUCATION = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']


@lru_cache(maxsize=1)
def typical_applicant():
    """Values for the fields the form does not ask for: a typical applicant (median or most common value in train).

    Without them those fields would be missing, which real applicants almost never are, and the model reads that as risk.
    """
    return json.loads((MODEL_DIR / 'typical_applicant.json').read_text(encoding='utf-8'))['values']


def score_form(age_years, years_employed, gender, education, contract_type, income, credit, annuity, goods_price, ext_1, ext_2, ext_3):
    """The form's fields on top of a typical applicant, with no credit history, scored by the same predict()."""
    days = lambda years: None if years is None else -int(round(float(years) * 365.25))
    application = {**typical_applicant(), 'SK_ID_CURR': 0, 'DAYS_BIRTH': days(age_years), 'DAYS_EMPLOYED': days(years_employed), 'CODE_GENDER': gender,
                   'NAME_EDUCATION_TYPE': education, 'NAME_CONTRACT_TYPE': contract_type, 'AMT_INCOME_TOTAL': income, 'AMT_CREDIT': credit,
                   'AMT_ANNUITY': annuity, 'AMT_GOODS_PRICE': goods_price, 'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3}
    result = predict(application)
    summary = f"Probability of payment difficulties: {result['probability']:.1%}\nDecision at {result['threshold']:.4f}: {result['decision'].upper()}"
    reasons = [[r['feature'], r['value'], r['odds_multiplier'], r['direction']] for r in result['reasons']]
    return summary, reasons


def build_ui():
    import gradio as gr

    return gr.Interface(
        fn=score_form,
        inputs=[
            gr.Number(label='Age (years)', value=45), gr.Number(label='Years employed', value=6),
            gr.Dropdown(['F', 'M'], label='Gender', value='F'), gr.Dropdown(EDUCATION, label='Education', value='Higher education'),
            gr.Dropdown(['Cash loans', 'Revolving loans'], label='Contract type', value='Cash loans'),
            gr.Number(label='Annual income', value=180000), gr.Number(label='Credit amount', value=600000),
            gr.Number(label='Annuity', value=27000), gr.Number(label='Price of the goods', value=500000),
            gr.Slider(0, 1, value=0.5, label='External score 1'), gr.Slider(0, 1, value=0.6, label='External score 2'),
            gr.Slider(0, 1, value=0.5, label='External score 3'),
        ],
        outputs=[gr.Textbox(label='Result'), gr.Dataframe(headers=['feature', 'value', 'odds multiplier', 'direction'], label='Main reasons')],
        title='Home Credit default risk',
        description=('Scores an applicant from a few application fields. Every other application field takes a typical value (the median, '
                     'or the most common answer, among the training applicants), and the applicant is scored as having no credit history: '
                     'no bureau records and no earlier applications. The API at /predict takes the full application and the history.'),
    )


try:
    import gradio as gr
    app = gr.mount_gradio_app(app, build_ui(), path='/ui')
except ImportError:                             # the API works without the form
    pass
