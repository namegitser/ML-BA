# Budget-Constrained Prospect Selection

This project compares three ways to rank customers for a costly campaign:

1. **RFM history:** rank by prior 12-month spend.
2. **Calibrated expected response:** calibrated visit probability under treatment,
   weighted by `log1p(history)`.
3. **Causal uplift:** an X-learner that ranks the estimated incremental visit
   probability from treatment.

The analysis uses Kevin Hillstrom's MineThatData randomized email experiment,
filtering to the `Womens E-Mail` and `No E-Mail` arms. The intervention in the
source data is an email, not a $5 catalog/voucher. The $5 contact cost is an
explicit scenario assumption, not a cost measured in the experiment. Reported
"net contribution" subtracts that cost from incremental **gross spend** and is
an optimistic upper bound on profit because it excludes cost of goods, voucher
redemption, and other operating costs.

## Run

Python 3.10 or later is required.

```powershell
python -m venv .env
.\.env\Scripts\python.exe -m pip install -r requirements.txt
.\.env\Scripts\python.exe prospect_selection.py
.\.env\Scripts\python.exe slides\build_slides.py
```

The checked-in `data\hillstrom.csv` is the original public dataset. To fetch a
fresh copy, run `prospect_selection.py --download`. Generated analysis tables,
figures, and `results.json` are written to `outputs\`; the presentation is
written to `slides\prospect_selection.pptx`.

Useful options:

```powershell
.\.env\Scripts\python.exe prospect_selection.py --help
.\.env\Scripts\python.exe slides\build_slides.py --help
```

## Method and interpretation

- Only the eight pre-campaign covariates listed in `prospect_selection.py` are
  used for ranking. Outcomes and treatment assignment are excluded from model
  features.
- A fixed, stratified 70/30 split holds out 12,808 customers. Model fitting,
  calibration, and X-learner pseudo-outcome construction use only the training
  split. The X-learner imputes counterfactual outcomes with a model trained on
  the opposite randomized arm, so each customer's pseudo-outcome is not
  generated from a model trained on that customer's observed outcome.
- The test-set rankings are fixed before examining test outcomes. Within each
  selected cohort, randomized treated/control differences estimate the causal
  effect for that cohort; finite-sample intervals are shown because these
  estimates can be noisy.
- Visit is the denser training outcome. All final economic comparisons use
  observed test-set spend as well as visit outcomes.
- Isotonic calibration is evaluated on treated customers in the held-out set.
  Calibration improves probability interpretation; it does not guarantee a
  better ranking or a profitable campaign.
- The spend-lift interval uses a stratified bootstrap (treated and control
  customers resampled separately). "Net contribution" is incremental gross
  spend less the assumed contact cost, not accounting profit.

## Outputs

- `outputs\results.json`: cohort, balance, calibration, shortlist, and
  uncertainty results used by the slides.
- `outputs\shortlist_metrics.csv`, `outputs\shortlist_profiles.csv`, and
  `outputs\covariate_balance.csv`: auditable tables.
- `outputs\calibration.png` and `outputs\qini.png`: presentation figures.
- `slides\prospect_selection.pptx`: eight-slide 16:9 presentation, built from
  the analysis outputs.

## Data source

Kevin Hillstrom, *MineThatData E-Mail Analytics and Data Mining Challenge*
(2008), public randomized experiment:
<http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv>
