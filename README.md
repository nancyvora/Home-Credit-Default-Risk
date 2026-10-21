# Home Credit Default Risk

[![CI](https://github.com/nancyvora/Home-Credit-Default-Risk/actions/workflows/ci.yml/badge.svg)](https://github.com/nancyvora/Home-Credit-Default-Risk/actions/workflows/ci.yml)

Predicting which loan applicants will have payment difficulties, from their application and their credit history:
[Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

The project goes from exploratory analysis of each of the six data tables to a tested, containerised scoring
service. Every modelling choice was made on cross-validation inside a fixed development set, and checked once at
the end on a holdout that had been set aside before any model was built.

## Results

| Model | Features | Cross-validated ROC AUC | Holdout ROC AUC | Kaggle private |
|---|---:|---:|---:|---:|
| Application table only (notebook 02) | 36 | 0.7685 | 0.7715 | 0.7638 |
| + credit bureau (04) | 60 | 0.7779 | | 0.7753 |
| + earlier Home Credit applications (06) | 85 | 0.7856 | | 0.7842 |
| + instalment payment record (08) | 108 | 0.7932 | | 0.7934 |
| + POS / cash loan balances (10) | no gain (+0.00007), left out | | | |
| **+ credit card use: final model (12, 13)** | **111** | **0.7948** | **0.7947** (0.7883 to 0.8014) | **0.7943** |

- **Three independent measurements agree.** From the application-only model to the final model: +0.026 in
  cross-validation, +0.023 on the holdout, +0.031 on Kaggle's private test set.
- **Calibrated probabilities.** On the holdout the model predicts an 8.12% default rate against 8.07% observed,
  with no tenth of predicted risk more than 0.73 points off.
- **For decisions.** At the reference threshold (a missed default costs ten times a wrongly declined applicant),
  the model declines 27.6% of holdout applicants and catches 68.3% of the defaults. The application-only model
  declined more (28.9%) and caught fewer (65.6%). The riskiest tenth of applicants holds 38.7% of all defaults.

## How the result was protected from leakage and luck

- **A frozen split.** 20% of train (61,503 applicants), stratified, was set aside before any model was built. It
  was opened twice: for the application-only model in notebook 02, and for the final model in notebook 13. Nothing
  was chosen from it either time. The code recomputes the split on every run and refuses to continue if it differs.
- **Features from the past only.** Every feature of an applicant uses only that applicant's own records dated
  before the application. The validation checks enforce this time rule on every raw table, and tests check that an
  applicant's features do not change when other applicants' records do.
- **Everything that learns is fitted inside the folds.** Category levels, calibration and feature selection are
  fitted on the training folds only (one scikit-learn pipeline).
- **One rule for every change.** A table, a feature, a setting or a number of trees was adopted only if it beat the
  current model by four conditions on the five folds: higher on average, higher in at least four folds, more than
  twice its standard error, and above a noise floor measured from seeds. The POS / cash table failed and was left
  out. So did a settings search winner: it passed under the search protocol, then lost at the final learning rate.
- **Kaggle scores were reported, never used to choose.**

## The model

LightGBM, 3,500 trees at a learning rate of 0.02, on 111 features. Platt calibration is fitted on out-of-fold
predictions inside the training rows. Each table's share of the model's total contribution:

| Table | Features | Share of contribution | Built in |
|---|---:|---:|---|
| Application | 36 | 52% | notebook 01 |
| Earlier Home Credit applications | 25 | 16% | notebook 05 |
| Instalment payment record | 23 | 16% | notebook 07 |
| Credit bureau | 24 | 14% | notebook 03 |
| Credit card balances | 3 | 2% | notebook 11 |

`bureau_balance.csv` was not used, by a decision taken at the start.

## Project structure

```
configs/             final_model.yaml (the recipe: features, settings, threshold), application_only.yaml (the reference)
notebooks/           01-13: EDA and features for each table, then the model with each table; 13: settings search and holdout
src/
  data/              load_data.py (raw and processed tables), preprocess.py (cleaning rules, frozen split)
  features/          build_features.py, plus one verified module per table (bureau, previous_application, installments, ...)
  models/            train.py, evaluate.py, tune.py (Optuna with the two-test rule), pipeline.py (shared by every notebook)
  serving/           inference.py, model/ (the final model, its metadata, input schema and holdout evaluation)
  app/               app.py (FastAPI + Gradio form), main.py (starts the server)
  utils/             validate_data.py (Great Expectations checks), utils.py (config, logging)
scripts/run_pipeline.py    load -> validate -> preprocess -> features -> evaluate -> train -> holdout -> submission
tests/               71 pytest tests
Dockerfile           the scoring service
```

## Running it

Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Download the competition files from Kaggle into `data/raw/` (`application_train.csv`, `bureau.csv`, ...).

**The whole pipeline, from the raw files** (about 30 to 45 minutes; logged to MLflow with `--mlflow`):

```bash
python scripts/run_pipeline.py
```

It reproduces the notebooks exactly: cross-validated ROC AUC 0.79476, holdout 0.7947, and a submission identical to
notebook 12's. Its outputs go to `artifacts/pipeline/`.

**Stages on their own:**

```bash
python -m src.utils.validate_data                 # check the raw files
python -m src.features.build_features             # build every feature table into data/processed
python -m src.models.evaluate                     # five-fold cross-validation of the recipe in configs/
python -m src.models.train --rows all             # fit on all of train and save the model
python -m src.models.tune                         # settings search (1 to 4 hours); never overwrites the config
python -m src.serving.inference                   # Kaggle submission from the saved feature tables
```

**The scoring service:**

```bash
python -m src.app.main                            # http://127.0.0.1:8000/docs, form at /ui
docker build -t home-credit-api . && docker run -p 8000:8000 home-credit-api
```

`POST /predict` takes one application row and, optionally, the applicant's records in each history table. It
returns the probability, the decision at the threshold and the five features that moved the risk most:

```json
{"application": {"SK_ID_CURR": 100001, "AMT_CREDIT": 568800, "AMT_ANNUITY": 20560.5, "EXT_SOURCE_2": 0.79, "EXT_SOURCE_3": 0.16},
 "history": {"bureau": [{"SK_ID_BUREAU": 5896630, "DAYS_CREDIT": -857, "CREDIT_ACTIVE": "Closed", "...": "..."}]}}
```

The features of a request are built by the same preprocessing and feature code as the training tables. Scoring
test applicants one at a time from their raw records gives exactly their batch scores; a test checks this.

**Tests:**

```bash
python -m pytest                                  # all 71 (about 3 minutes)
SKIP_DATA_TESTS=1 python -m pytest                # the 64 that need no Kaggle files, as in CI
```

## Known limits

- **Kaggle's test applicants are a later population.** Scores there run slightly below cross-validation (0.0005
  for the final model), as expected.
- **Calibration is good on average and by tenth, but not in every group.** Applicants with an unpaid instalment on
  record (about 0.16%) are under-predicted: 19.7% predicted against 27.2% observed, though they are ranked correctly.
- **The threshold is a business choice.** The reference threshold assumes a missed default costs ten times a wrongly
  declined applicant; thresholds for cost ratios of 5 and 20 are in the config.
- **A request takes about 2 seconds**, mostly building the five history tables for one applicant. Fine for a
  demonstration; worth optimising before real traffic.
