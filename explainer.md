# Prospect Selection: Problem, Methods, Evidence, and Decision

## Executive summary

This project asks a decision question, not simply a prediction question:

> Given a fixed budget to contact at most 1,500 of 12,808 eligible customers, which customers should receive the campaign?

The distinction matters. A customer who is likely to buy anyway may be a good prediction target but a poor use of a costly offer. The desired target is a customer whose behavior changes because of the campaign, and whose incremental value is large enough to cover the contact cost.

We use Kevin Hillstrom's randomized MineThatData email experiment, comparing the `Womens E-Mail` arm with `No E-Mail`. We train models on pre-campaign customer information, rank a held-out test group without using its outcomes, and then compare treated and control outcomes among each locked top-1,500 list.

The results support a cautious conclusion:

- The X-learner shortlist has the largest observed **visit** lift at K=1,500: **+9.47 percentage points**.
- It does not have the largest observed **spend** lift: its estimate is **+$0.64 per selected customer**, with a 95% bootstrap interval of **−$0.36 to +$2.25**.
- No shortlist reaches the assumed **$5 per-contact** break-even amount, even at the upper end of its spend-lift interval. RFM has the best spend point estimate at **+$2.00 per customer**, with an interval of **−$0.05 to +$4.51**.
- The experiment was an email campaign, not a $5 catalog/voucher campaign. The $5 cost is a scenario assumption, and the spend calculation is gross revenue rather than contribution margin. The results therefore support a **no-go for scaling the hypothetical $5 campaign on this evidence**, not a universal conclusion about all campaigns.

The analysis and generated numbers are in [`prospect_selection.py`](./prospect_selection.py) and [`outputs/results.json`](./outputs/results.json). The presentation is [`slides/prospect_selection.pptx`](./slides/prospect_selection.pptx). The original project requirements remain in [`mlba_prospect_selection_project_playbook.md`](./mlba_prospect_selection_project_playbook.md).

---

## 1. The business problem

### The decision

The hypothetical retailer has:

| Quantity | Value |
|---|---:|
| Eligible prospects in the held-out cohort, N | 12,808 |
| Maximum number to contact, K | 1,500 |
| Share of the test cohort selected | 11.71% |
| Assumed cost per contact, c | $5 |
| Assumed total contact cost at K=1,500 | $7,500 |

The operational task is to create a ranked list. The economic task is to make the incremental value of that list exceed its cost.

### Why ordinary response prediction can make the wrong choice

Let `Y(1)` be a customer's outcome if contacted and `Y(0)` the outcome if not contacted. A response model trained on contacted customers tries to estimate:

$$
\mu_1(x) = E[Y(1)\mid X=x]
$$

That answers, “How likely is this customer to respond if contacted?” It does **not** answer, “How much more likely are they to respond because they were contacted?”

A high-response customer might also have a high chance of visiting or buying without the campaign. Contacting that “sure thing” can spend the budget without causing an incremental outcome. The causal quantity of interest is:

$$
\tau(x) = E[Y(1)-Y(0)\mid X=x] = \mu_1(x)-\mu_0(x)
$$

where `μ0(x)` is the expected outcome without contact. In the campaign setting, a positive `τ(x)` indicates predicted persuasion; a negative value warns that treatment may reduce the outcome. These are averages conditional on observed features—not proof of any one person's unobservable counterfactual.

### The economic decision rule

For a selected group of K customers, let `ΔS@K` be the incremental spend per selected customer, estimated by the treated-versus-control spend difference in that group. The project's simplified economics are:

$$
\widehat{\text{Incremental gross spend}}@K = K \times \widehat{\Delta S}@K
$$

$$
\widehat{\text{Net contribution proxy}}@K
  = K \times (\widehat{\Delta S}@K-c)
$$

The break-even condition under this simplified calculation is `ΔS@K > c`. With `c=$5`, incremental **gross spend** must exceed $5 per contacted customer just to cover the assumed contact cost. Because the dataset does not provide product margin, voucher cost, fulfillment cost, or long-term customer value, this is not accounting profit. It is an optimistic gross-spend proxy.

---

## 2. What we test, and why this dataset helps

### Hypotheses under comparison

The project compares four policies:

1. **Random selection:** a budget-neutral reference point.
2. **RFM history:** rank by prior 12-month spend.
3. **Calibrated expected-value proxy:** estimate visit probability under treatment and weight it by a transformed historical-spend feature.
4. **X-learner uplift:** rank by estimated incremental visit probability due to treatment.

The central test is whether a causal-uplift ranking selects a group with greater incremental outcomes than a response/value heuristic or a simple RFM baseline—and whether those incremental outcomes are enough to support the $5 contact cost.

The expected outcome is not that every sophisticated model must win. These are competing hypotheses. In particular:

- Visit is a denser signal to learn from than purchase or spend.
- RFM may select customers with high historical and future spend, but not necessarily the most persuadable customers.
- Uplift may improve visit targeting without improving dollars. That would expose a mismatch between the training proxy and the business objective.
- Calibration may make probabilities more interpretable, but it does not guarantee a lower Brier score, a better ranking, or positive economics.

### Why use a randomized experiment

The original dataset contains 64,000 customers across three randomized email groups. This analysis filters to:

- `Womens E-Mail` as treatment, **21,387 customers**
- `No E-Mail` as control, **21,306 customers**

The resulting two-arm cohort has **42,693 customers**. The men's email arm is excluded to keep the intervention definition focused: the causal comparison here is women's email versus no email, not a mixture of two different campaign treatments.

In a randomized controlled trial (RCT), treatment assignment is independent of customers' pre-treatment characteristics in expectation. This makes the treatment-control outcome difference within a group defined using pre-treatment features an unbiased estimate of the treatment effect for that selected group, subject to sampling error and the experiment's design.

For an observational campaign log, high-value customers might be more likely to receive an offer. Then `T` is associated with customer characteristics, and a simple treated-control comparison can be confounded. Propensity weighting or doubly robust causal methods, plus overlap checks, would be needed; this project does not claim that the RCT pipeline can be transferred unchanged to observational data.

### Outcomes and observed base rates

The data records outcomes over the two weeks after the email:

| Outcome | No E-Mail control | Womens E-Mail treatment |
|---|---:|---:|
| Website visit | 10.62% | 15.14% |
| Purchase / conversion | 0.57% | 0.88% |
| Average gross spend per customer | $0.65 | $1.08 |
| Customers with zero spend | 99.43% | 99.12% |

The overall visit lift is **+4.52 percentage points**. The overall spend lift is approximately **+$0.42 per customer**. At a hypothetical $5 contact cost, that population-average spend is far below break-even: about `0.42 - 5 = -$4.58` per contact before cost of goods or other expenses.

Spend is very sparse and right-skewed; the largest observed spend is $499. That motivates using visit as the model-training proxy in this build while still evaluating the final chosen groups on observed spend. It does **not** make visit equivalent to revenue. The results explicitly demonstrate why the distinction matters.

---

## 3. Features, integrity checks, and the holdout

### Pre-campaign inputs

Only the following eight customer fields are used as model inputs:

| Field | Role / meaning |
|---|---|
| `recency` | Months since prior purchase |
| `history` | Prior 12-month spend |
| `history_segment` | Historical-spend tier |
| `mens` | Prior men's-merchandise purchase indicator |
| `womens` | Prior women's-merchandise purchase indicator |
| `newbie` | New-customer indicator |
| `zip_code` | Urban/suburban/rural category |
| `channel` | Phone/web/multichannel category |

They are treated as pre-decision information based on the dataset's definitions. Post-campaign fields—`visit`, `conversion`, and `spend`—are outcomes, not model features. `segment` is used to define treatment, not as a predictor.

The pipeline passes the five numeric/binary fields through and one-hot encodes `history_segment`, `zip_code`, and `channel`. This lets the tree models use category levels without treating category names as ordered numbers. The encoder is fit on the training data and ignores unknown test-time categories.

### Randomization balance audit

The script compares the treatment and control groups on the full two-arm cohort using standardized mean differences:

$$
\text{SMD} =
\frac{\bar X_1-\bar X_0}
{\sqrt{(s_1^2+s_0^2)/2}}
$$

For categorical fields, it computes an SMD for each one-hot category and summarizes the field by its largest absolute category-level SMD. All **8 of 8** pre-campaign fields are below the common **0.10** diagnostic threshold; the largest is **0.014**.

This is reassuring evidence that the observed covariates are balanced. It is not a test of every possible unobserved variable, and balance alone does not establish randomization; the experiment's randomized assignment is the basis for causal identification.

### Training and test split

The cohort is split once with `test_size=0.30`, `random_state=42`, stratified on the combination of treatment assignment and visit outcome:

- Training: **29,885**
- Held-out test: **12,808**

Stratifying preserves the treatment and visit proportions across the two partitions, including the relatively uncommon visit outcome. Visit is used to create the split, but it is never included in the feature matrix or used to rank test customers. Models, calibrators, and uplift regressors are fit using the training partition only. Test outcomes are opened for evaluation after each top-K list is formed.

This creates an out-of-sample policy evaluation for the specified random split and fitted models. It is not repeated cross-validation over many seeds, so results can still depend on this split and training run.

---

## 4. The three ranking approaches

All model-based approaches use HistGradientBoosting models with fixed settings. The classifier uses 120 boosting iterations, at most 15 leaves per tree, a 0.07 learning rate, and L2 regularization of 1.0. These settings provide a practical nonlinear model for interactions without adding a separate LightGBM dependency. They were not selected by a broad hyperparameter search; do not describe them as the proven optimal configuration.

### A. RFM historical-spend baseline

The baseline score is simply:

$$
s_{\text{RFM}}(x_i)=\text{history}_i
$$

It is deliberately simple, transparent, and easy to explain. It answers whether a common business rule—contact the historical high spenders—does well enough to challenge more complex models. It is **not** a causal model: past spend may predict future spend, but it does not measure the effect of this campaign.

### B. Calibrated expected-value proxy

The pipeline fits a visit classifier on the **treated training arm**, estimating:

$$
\hat p_1(x)=\widehat P(\text{visit}=1\mid X=x,T=1)
$$

It then ranks test customers by:

$$
s_{\text{EV}}(x_i)=\hat p_{1,\text{cal}}(x_i)\times\log(1+\text{history}_i)
$$

The logarithm preserves the ordering of historical spend while reducing the influence of very large past-spend values. The product combines expected response under treatment with a historical-value proxy.

This is called a **calibrated expected-value proxy**, not a literal expected dollar-revenue forecast. It multiplies visit probability by transformed historical spend; it does not estimate a customer's future order value or profit. It answers, “Who looks likely to visit if treated and has a history suggesting value?” It can still over-rank “sure things.”

The treated and control visit classifiers are calibrated with five-fold isotonic calibration within the training data. Isotonic regression learns a flexible monotone mapping from classifier scores to observed frequencies. The held-out Brier score is:

$$
\text{Brier}=\frac{1}{n}\sum_i(\hat p_i-y_i)^2
$$

Lower is better. On treated test customers, the raw score is **0.12519** and the isotonic score is **0.12547**. The calibrated score is slightly worse on this split. Thus the reliability plot is an audit, not a claim that calibration improved this result. This implementation uses isotonic calibration; it does not report a Platt-scaling comparison.

There is a practical comparison caveat: the raw score comes from one classifier fit on all treated training rows, while `CalibratedClassifierCV(cv=5)` builds calibrated fold models. The Brier comparison therefore evaluates the raw and calibrated **pipelines**, not an isolated calibration transform applied to identical base-model predictions.

### C. X-learner causal uplift

The project estimates separate response surfaces in the two randomized training arms:

$$
\hat\mu_1(x)\approx P(\text{visit}=1\mid X=x,T=1), \qquad
\hat\mu_0(x)\approx P(\text{visit}=1\mid X=x,T=0)
$$

The X-learner forms pseudo-effects using the other arm's response model:

$$
\tilde D_i^{(1)}=Y_i-\hat\mu_0(X_i), \quad i:T_i=1
$$

$$
\tilde D_i^{(0)}=\hat\mu_1(X_i)-Y_i, \quad i:T_i=0
$$

The first pseudo-effect estimates how a treated customer's observed outcome differs from the control-model counterfactual. The second estimates how the treated-model counterfactual differs from a control customer's observed outcome. Each pseudo-outcome uses a counterfactual predictor trained on the opposite randomized arm, rather than a model trained on that customer's observed outcome.

Two HistGradientBoosting regressors learn the pseudo-effects, `τ̂1(x)` and `τ̂0(x)`. They are blended using the training treatment fraction `e`:

$$
\hat\tau_X(x)=e\hat\tau_0(x)+(1-e)\hat\tau_1(x)
$$

Here `e` is approximately 0.501, as expected from the nearly balanced RCT. The final score is the estimated treatment effect on **visit probability**, and customers are sorted from highest to lowest.

The RCT supports the causal interpretation of average effects, but a CATE model is still an estimate. It does not reveal the true individual pair `(Y(1),Y(0))`; only one outcome is observed for each customer. Regularization, model error, and finite sample size can make the ranking noisy.

### What is and is not included

The delivered shortlist comparison includes random selection, RFM, calibrated EV, and the X-learner. It fits the arm-specific response models needed by the X-learner, but does not report a separate T-learner shortlist result, spend-based uplift model, or Platt calibration result. Those are possible extensions, not findings from this run.

---

## 5. How the held-out evaluation works

### Lock the list before reading outcomes

For each approach, the score is computed using only the 12,808 test customers' pre-campaign features and training-fitted model. The first 1,500 indices in descending score order form the shortlist. The random baseline uses a seeded random score.

Only after selection, the pipeline separates the selected customers into their randomized treatment and control arms. For a test outcome `Y`, the selected-group lift is:

$$
\widehat{\text{Lift}}@K =
\frac{1}{n_T(K)}\sum_{i\in S_K,T_i=1}Y_i
-
\frac{1}{n_C(K)}\sum_{i\in S_K,T_i=0}Y_i
$$

Because `S_K` is determined from pre-treatment features and training-fitted scores—not the test outcomes—randomization remains the reason this difference estimates the selected group's treatment effect. The held-out sample has about 750 customers in each arm per shortlist, so the estimate has meaningful sampling uncertainty.

The pipeline computes this comparison for **visit** and **spend**. Visit intervals use a normal approximation to the difference of two sample means. Spend intervals use 2,000 percentile-bootstrap replicates, resampling treated and control spend separately within the fixed shortlist. These intervals are conditional on the fitted model, selected list, and split; they do not include model-training variation across new splits or random seeds.

### Qini curve

The Qini curve sorts the test set by each model score and, at every targeting depth, compares the cumulative treated visits to the control visits scaled for the number of people in each arm:

$$
Q(k)=
\sum_{i\le k,T_i=1}Y_i
-
\left(\sum_{i\le k,T_i=0}Y_i\right)
\frac{N_T(k)}{N_C(k)}
$$

It visualizes how much incremental **visit** response a ranking accumulates as the budget grows. The code includes a random-targeting reference and marks K=1,500. The curve is not a dollar-profit curve, and this run does not report a Qini AUC or a confidence band for the curve.

---

## 6. Results at the budget cap

The full machine-readable results are in [`outputs/results.json`](./outputs/results.json), with shortlist tables in [`outputs/shortlist_metrics.csv`](./outputs/shortlist_metrics.csv).

| Ranking policy | Visit lift in top 1,500 | Spend lift / selected customer | 95% bootstrap interval for spend lift | Net contribution proxy at $5 |
|---|---:|---:|---:|---:|
| Random selection | +5.06 pp | +$0.27 | −$1.26 to +$1.61 | −$7,091 |
| RFM history | +7.06 pp | +$2.00 | −$0.05 to +$4.51 | −$4,506 |
| Calibrated expected value | +6.21 pp | +$0.84 | −$0.66 to +$2.69 | −$6,241 |
| X-learner uplift | **+9.47 pp** | +$0.64 | −$0.36 to +$2.25 | −$6,538 |

`pp` means percentage points. The net contribution proxy is `1,500 × (spend lift per person − $5)`. For RFM, the estimated incremental gross spend is about **$2,994** and the assumed contact cost is **$7,500**, giving about **−$4,506**. Its spend-lift interval implies a net-proxy interval of approximately **−$7,570 to −$735**. Even the upper end is below break-even; the other three spend-lift upper bounds are also below $5.

The visit result and the spend result answer different questions. X-learner uplift is strongest for visits at this K, but the observed spend difference does not show that those extra visits pay for the contact. This is not a contradiction: an incremental visit may not become a purchase, and a visit-optimized learner is not trained to maximize dollars.

The calibrated-EV and X-learner top-1,500 lists overlap by **628 customers**, or **41.9% of K**; their Jaccard similarity is **26.5%**. The playbook's `<25% overlap` was an expectation, not a guaranteed property. The observed overlap is higher by the simple-overlap measure, and this run should report the measured value rather than retrofit the expectation.

The profiles also differ. For example, average historical spend is about **$783** in RFM's list versus **$409** in the X-learner list. The X-learner list has **97.9%** prior women's-category buyers versus **68.4%** for RFM. These are descriptive differences in selected lists; they do not prove that customers are literally “persuadables,” “sure things,” or “sleeping dogs.”

---

## 7. Why each choice is defensible—and its limit

| Choice | Why it is defensible | What it does not establish |
|---|---|---|
| Womens email vs no email | A clean binary intervention/control contrast in a randomized experiment. | Effects of men's email, a catalog, a voucher, or a different channel. |
| Eight pre-campaign features | Mirrors information available when targeting must happen; prevents using post-campaign outcomes as predictors. | Absence of omitted customer characteristics or all forms of leakage in a different data source. |
| Stratified 70/30 split | Gives a reproducible training/holdout separation and preserves treatment/visit proportions. | Robustness across all possible splits or future populations. |
| RFM baseline | Transparent business benchmark; shows what a simple rule achieves. | Causal uplift or future customer lifetime value. |
| Calibrated EV proxy | Tests response likelihood weighted by an observed historical-value proxy. | True expected revenue/profit; calibration did not lower Brier on this holdout. |
| X-learner on visit | Uses both randomized arms to rank estimated incremental visits; visit is much denser than spend. | Individual treatment effects, or a guarantee that visit uplift means dollar uplift. |
| Held-out treated-control differences | Uses randomized assignments inside lists fixed from pre-treatment data to estimate subgroup lift. | Zero uncertainty; observed effects in future campaigns; corrections for model-training uncertainty. |
| Spend lift minus $5 | Directly tests the scenario's simple per-contact break-even condition. | Accounting profit, because gross margin and campaign-specific costs are unavailable. |

The cost assumption is intentionally set to **$5**, matching the decision statement in the project playbook. It is not measured by the Hillstrom dataset. It also differs from the `$1.50` value in the sample script embedded in that playbook; the implemented project uses `$5` so that the analysis matches the stated business scenario.

---

## 8. Limitations and the next defensible experiment

1. **External validity:** The source treatment is an email campaign from 2008. A high-touch printed catalog plus voucher may change response, cost, and treatment effects.
2. **Gross spend is not margin:** The data does not provide product cost, voucher redemption cost, returns, fulfillment, or other variable costs. A positive gross-spend estimate would still not prove positive profit.
3. **Short horizon:** Outcomes cover two weeks. Retention, repeat purchase, discount effects, and longer-term customer value are not modeled.
4. **Proxy mismatch:** Ranking is trained on visits, but the business decision is monetary. A future model should estimate incremental contribution margin directly if enough data exists, or combine a causal response estimate with validated margin/value estimates.
5. **Sparse financial outcome:** Spend is zero for over 99% of customers and has a long right tail. The bootstrap interval reflects uncertainty in the observed fixed shortlist, but not every source of model uncertainty.
6. **Single split and fixed model choices:** The result is reproducible with seed 42, but stability across seeds, model specifications, K values, and treatment campaigns should be studied before deployment.
7. **No individual counterfactual labels:** RCTs identify average and subgroup effects in expectation. They cannot reveal which specific customer would have purchased under both treatment states.
8. **Policy comparisons are exploratory:** The reported intervals are per-shortlist intervals, not simultaneous intervals adjusted for comparing four policies. The largest observed point estimate should not be treated as a statistically proven winner.

The next step should be a small randomized pilot of the **actual** offer and channel. Record contribution margin after discount and fulfillment, preserve a no-contact holdout, predefine the targeting policy and K, and estimate uncertainty before scaling. A sensible scale gate is based on incremental contribution after all variable costs—not visits or gross revenue—and should specify in advance how much uncertainty the business is willing to accept.

---

## 9. Reproducing the analysis

From the project directory on Windows:

```powershell
.\.env\Scripts\python.exe prospect_selection.py
.\.env\Scripts\python.exe slides\build_slides.py
```

The analysis writes:

- `outputs\results.json` — cohort facts, calibration scores, shortlist results, and profiles
- `outputs\shortlist_metrics.csv` — visit/spend lifts, intervals, and cost-adjusted proxy
- `outputs\covariate_balance.csv` — numeric and one-hot balance diagnostics
- `outputs\calibration.png`, `outputs\qini.png`, `outputs\balance.png` — exhibits used in the deck
- `slides\prospect_selection.pptx` — the eight-slide presentation

The dataset is [`data/hillstrom.csv`](./data/hillstrom.csv), from Kevin Hillstrom's *MineThatData E-Mail Analytics and Data Mining Challenge* (2008). The pipeline can also download the public CSV with `--download`; default values and optional arguments are described by `prospect_selection.py --help`.
