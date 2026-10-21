"""The Gradio interface: a quick form, a full request in JSON, and what the model is.

Used in two places: mounted at ``/ui`` by the FastAPI app (``src/app/app.py``), and on its own as the Hugging Face
Space (``deploy/huggingface/app.py``). Every score goes through ``src/serving/inference.predict``, the same code as
the API and the batch submission.
"""
import json
from functools import lru_cache
from pathlib import Path

from src.serving.inference import HISTORY_TABLES, MODEL_DIR, load_model, predict

EDUCATION = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']
EXAMPLE_REQUEST = Path(__file__).resolve().parent / 'example_request.json'
REASON_HEADERS = ['feature', 'value', 'odds multiplier', 'direction']


@lru_cache(maxsize=1)
def typical_applicant():
    """Values for the fields the form does not ask for: a typical applicant (median or most common value in train).

    Without them those fields would be missing, which real applicants almost never are, and the model reads that as risk.
    """
    return json.loads((MODEL_DIR / 'typical_applicant.json').read_text(encoding='utf-8'))['values']


def _summary(result):
    return (f"Probability of payment difficulties: {result['probability']:.1%}\n"
            f"Decision at the threshold {result['threshold']:.4f}: {result['decision'].upper()}")


def _reasons(result):
    return [[r['feature'], r['value'], r['odds_multiplier'], r['direction']] for r in result['reasons']]


def score_form(age_years, years_employed, gender, education, contract_type, income, credit, annuity, goods_price, ext_1, ext_2, ext_3):
    """The form's fields on top of a typical applicant, with no credit history, scored by the same predict()."""
    days = lambda years: None if years is None else -int(round(float(years) * 365.25))
    application = {**typical_applicant(), 'SK_ID_CURR': 0, 'DAYS_BIRTH': days(age_years), 'DAYS_EMPLOYED': days(years_employed), 'CODE_GENDER': gender,
                   'NAME_EDUCATION_TYPE': education, 'NAME_CONTRACT_TYPE': contract_type, 'AMT_INCOME_TOTAL': income, 'AMT_CREDIT': credit,
                   'AMT_ANNUITY': annuity, 'AMT_GOODS_PRICE': goods_price, 'EXT_SOURCE_1': ext_1, 'EXT_SOURCE_2': ext_2, 'EXT_SOURCE_3': ext_3}
    result = predict(application)
    return _summary(result), _reasons(result)


def score_json(text):
    """A full request, as the API takes it: {"application": {...}, "history": {"bureau": [...], ...}}."""
    try:
        request = json.loads(text)
        result = predict(request['application'], request.get('history') or {}, threshold=request.get('threshold'))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return f'Could not score this request: {e}', [], {}
    return _summary(result), _reasons(result), result


def about():
    meta = load_model().metadata
    evaluation_path = MODEL_DIR / 'holdout_evaluation.json'
    ev = json.loads(evaluation_path.read_text(encoding='utf-8')) if evaluation_path.exists() else {}
    hold, cv, kaggle = ev.get('holdout', {}), ev.get('cross_validation', {}), ev.get('kaggle', {})
    return f"""
**What it predicts.** The probability that a loan applicant will have payment difficulties (Kaggle: Home Credit Default
Risk), from the application and the applicant's credit history: credit bureau records, earlier Home Credit applications,
instalment payments and credit card balances.

**The model.** LightGBM with {len(meta['features'])} features, {meta['training']['trees']:,} trees, Platt calibration; trained
on {meta['training']['rows']:,} applicants.

| Measured on | ROC AUC |
|---|---|
| Cross-validation, 5 folds of the development set | {cv.get('roc_auc', float('nan')):.4f} |
| Holdout, 61,503 applicants set aside before any model was built | {hold.get('roc_auc', float('nan')):.4f} (95% interval {hold.get('interval', [0, 0])[0]:.4f} to {hold.get('interval', [0, 0])[1]:.4f}) |
| Kaggle private leaderboard | {kaggle.get('private', float('nan')):.4f} |
| Application-only model, same holdout | {hold.get('application_only_roc_auc', float('nan')):.4f} |

**The decision.** Decline at a probability of {meta['decision']['reference_threshold']:.4f} or more: this assumes a missed
default costs ten times a wrongly declined applicant. On the holdout the model declines 27.6% of applicants and catches 68.3%
of the defaults; its average prediction is 8.12% against 8.07% observed.

**The reasons.** Each score lists the features that moved the risk most, as odds multipliers: above 1 raises the risk,
below 1 lowers it. They are the model's exact contributions (TreeSHAP), scaled by the calibration.

**Limits.** A demonstration, not a lending decision. The quick form scores an applicant with no credit history, and gives
every field it does not ask for a typical value. Applicants with an unpaid instalment on record are under-predicted
(about 20% predicted against 27% observed), though ranked correctly.

Code, notebooks and tests: [github.com/nancyvora/Home-Credit-Default-Risk](https://github.com/nancyvora/Home-Credit-Default-Risk)
"""


def build_ui(wrap=None):
    """The interface. ``wrap`` is applied to the scoring functions; the Hugging Face Space passes ``spaces.GPU``."""
    import gradio as gr

    wrap = wrap or (lambda f: f)
    form_fn, json_fn = wrap(score_form), wrap(score_json)
    example = EXAMPLE_REQUEST.read_text(encoding='utf-8') if EXAMPLE_REQUEST.exists() else '{"application": {"SK_ID_CURR": 1}}'
    with gr.Blocks(title='Home Credit default risk') as demo:
        gr.Markdown('# Home Credit default risk\nProbability that a loan applicant will have payment difficulties, with the reasons.')
        with gr.Tab('Quick check'):
            gr.Markdown('A few application fields. Every other field takes a typical value, and the applicant is scored as having '
                        'no credit history (no bureau records, no earlier applications).')
            with gr.Row():
                with gr.Column():
                    age = gr.Number(label='Age (years)', value=45)
                    employed = gr.Number(label='Years employed', value=6)
                    gender = gr.Dropdown(['F', 'M'], label='Gender', value='F')
                    education = gr.Dropdown(EDUCATION, label='Education', value='Higher education')
                    contract = gr.Dropdown(['Cash loans', 'Revolving loans'], label='Contract type', value='Cash loans')
                with gr.Column():
                    income = gr.Number(label='Annual income', value=180000)
                    credit = gr.Number(label='Credit amount', value=600000)
                    annuity = gr.Number(label='Annuity', value=27000)
                    goods = gr.Number(label='Price of the goods', value=500000)
                with gr.Column():
                    ext_1 = gr.Slider(0, 1, value=0.5, label='External score 1')
                    ext_2 = gr.Slider(0, 1, value=0.6, label='External score 2')
                    ext_3 = gr.Slider(0, 1, value=0.5, label='External score 3')
            button = gr.Button('Score', variant='primary')
            result = gr.Textbox(label='Result', lines=2)
            reasons = gr.Dataframe(headers=REASON_HEADERS, label='Main reasons')
            button.click(form_fn, [age, employed, gender, education, contract, income, credit, annuity, goods, ext_1, ext_2, ext_3], [result, reasons])
        with gr.Tab('Full application (JSON)'):
            gr.Markdown(f'The request the API takes: one application row, and the applicant\'s records in any of the history tables '
                        f'({", ".join(HISTORY_TABLES)}), with the columns of the Kaggle files. A field left out is treated as missing. '
                        'The example is made up.')
            request = gr.Code(value=example, language='json', label='Request')
            button_json = gr.Button('Score', variant='primary')
            result_json = gr.Textbox(label='Result', lines=2)
            reasons_json = gr.Dataframe(headers=REASON_HEADERS, label='Main reasons')
            raw_json = gr.JSON(label='Response')
            button_json.click(json_fn, request, [result_json, reasons_json, raw_json])
        with gr.Tab('About the model'):
            gr.Markdown(about())
    return demo
