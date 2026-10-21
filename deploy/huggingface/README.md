---
title: Home Credit Default Risk
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Loan default probability with the reasons (LightGBM)
---

# Home Credit default risk

The probability that a loan applicant will have payment difficulties, with the features that moved the risk most.
LightGBM on 111 features from the application, credit bureau records, earlier Home Credit applications, instalment
payments and credit card balances (Kaggle: Home Credit Default Risk).

| Measured on | ROC AUC |
|---|---|
| Cross-validation, development set | 0.7948 |
| Holdout, 61,503 applicants set aside before any model was built | 0.7947 |
| Kaggle private leaderboard | 0.7943 |

Code, notebooks, tests and the API: [github.com/nancyvora/Home-Credit-Default-Risk](https://github.com/nancyvora/Home-Credit-Default-Risk)

A demonstration, not a lending decision.
