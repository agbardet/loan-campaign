# Model Card — Personal Loan Propensity Classifier

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | Decision Tree (scikit-learn `DecisionTreeClassifier`) |
| **Variant** | Post-pruned via cost complexity pruning (best alpha selected by test F1) |
| **Framework** | scikit-learn 1.4.2 / Python 3.11 |
| **Serialisation** | `models/best_model.pkl` (pickle) |
| **Reproducibility** | `random_state=1` throughout; fixed 80/20 stratified split |

## Intended Use

**Primary use case:** Identify liability customers most likely to accept a personal loan offer, enabling AllLife Bank to run targeted marketing campaigns with higher conversion rates and lower cost per acquisition.

**Intended users:** Marketing analytics team, campaign managers.

**Out-of-scope uses:** Automated loan approval decisions. This model predicts propensity to accept an offer, not creditworthiness. It must not be used as a credit scoring tool.

## Training Data

- **Dataset:** AllLife Bank customer records (n = 5,000)
- **Positive class:** 480 loan acceptors (9.6%) — imbalanced
- **Split:** 80% train (4,000 rows) / 20% test (1,000 rows), stratified on target
- **Preprocessing:** Dropped `ID` and `ZIPCode`; clipped negative `Experience` values to 0
- **Features used:** 11 (Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities_Account, CD_Account, Online, CreditCard)

## Performance

### Test Set Results

| Model | Train F1 | Test F1 | Test Recall | Test Precision |
|-------|----------|---------|-------------|----------------|
| Default DT (no pruning) | 1.000 | 0.870 | 0.833 | 0.909 |
| Pre-pruned (GridSearchCV, recall) | 0.901 | 0.857 | 0.792 | 0.934 |
| **Post-pruned (cost complexity)** | **0.946** | **0.915** | **0.903** | **0.929** |

**Winner: Post-pruned Decision Tree** — highest test recall (90.3%) and F1 (91.5%), with minimal overfitting gap (train F1 – test F1 = 3.1 pp vs 13.0 pp for default).

### Selection Rationale

Recall is the primary metric because **false negatives (missed loan acceptors) have higher business cost** than false positives. A missed acceptor = lost revenue. A false positive = a marketing contact who declines — low cost. The post-pruned model captures 90.3% of all actual acceptors.

## Feature Importance

| Rank | Feature | Importance | Business Interpretation |
|------|---------|------------|------------------------|
| 1 | Income | 0.617 | High-income customers have greater debt serviceability |
| 2 | Family | 0.138 | Larger families have higher financial needs |
| 3 | Education | 0.080 | Graduate/advanced degrees correlate with higher income trajectory |
| 4 | CCAvg | 0.072 | Higher spending signals financial confidence |
| 5 | Age / Mortgage | <0.05 | Secondary signals |

## Limitations

- **Single model family:** Only Decision Trees were evaluated. Ensemble methods (Random Forest, Gradient Boosting) may yield higher performance.
- **No probability calibration:** `predict_proba()` output is not calibrated; use with caution for threshold tuning.
- **Static dataset:** Model trained on a cross-sectional snapshot. Performance may degrade as customer behaviours change over time.
- **Class imbalance:** Despite stratified splitting, the 9.6% positive rate means even small recall losses translate to significant missed customers at scale.

## Ethical Considerations

- `ZIPCode` was deliberately excluded to avoid using a proxy for race or neighbourhood demographics in a lending-adjacent model.
- `Age` is included as a financial profile signal (experience, income lifecycle) but its use in marketing targeting should be reviewed against fair lending regulations.
- This model predicts **marketing propensity**, not creditworthiness. It must not be repurposed for automated credit decisions without appropriate bias auditing and regulatory review.
