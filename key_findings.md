# Key Findings — AllLife Bank Personal Loan Campaign

## Executive Summary

AllLife Bank sought to grow its loan portfolio by converting existing liability (deposit) customers into personal loan holders. Using a dataset of 5,000 customers, this analysis built a machine-learning classifier to identify which customers are most likely to accept a loan offer. The resulting post-pruned Decision Tree achieves **90.3% recall** and **91.5% F1 on the test set** — meaning the model correctly identifies 9 out of 10 likely loan acceptors, giving marketing teams a highly actionable targeting list.

---

## Customer Profile of Loan Acceptors

The 480 customers who accepted a loan offer (9.6% of the dataset) share a distinct profile:

- **Income:** Acceptors earn significantly more than non-acceptors (median ~$114K vs ~$66K). Income is the single strongest predictor in the model (importance: 61.7%).
- **Education:** Customers with graduate or advanced/professional degrees accept at roughly 3× the rate of undergraduates.
- **Family size:** Customers with families of 3–4 members accept at higher rates, likely reflecting greater borrowing needs (mortgages, education, childcare costs).
- **Credit card spend:** Higher average monthly spend (CCAvg) correlates with loan acceptance — an indicator of financial confidence and credit engagement.
- **CD Accounts:** Customers with certificate of deposit accounts are disproportionately represented among acceptors, suggesting they are more engaged with bank products overall.

---

## Key Predictors

The post-pruned Decision Tree surfaces a clear hierarchy:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Income | 61.7% |
| 2 | Family | 13.8% |
| 3 | Education | 8.0% |
| 4 | CCAvg (credit card spend) | 7.2% |
| 5+ | Age, Mortgage, CD_Account | <5% each |

The top 4 features alone account for ~90% of predictive power. A simple targeting rule derived from the tree: **customers earning above ~$106K per year with a family size ≥ 3 accept loans at a rate exceeding 70%** — far above the 9.6% baseline.

---

## Model Development & Selection

Three Decision Tree variants were evaluated:

| Model | Test F1 | Test Recall | Overfitting (Train–Test F1) |
|-------|---------|-------------|---------------------------|
| Default DT | 0.870 | 0.833 | 13.0 pp |
| Pre-pruned (GridSearchCV) | 0.857 | 0.792 | 4.4 pp |
| **Post-pruned (cost complexity)** | **0.915** | **0.903** | **3.1 pp** |

The default tree overfits severely (perfect training F1). Pre-pruning via GridSearchCV reduces overfitting but at the cost of recall. Post-pruning via cost complexity pruning finds the optimal alpha that maximises test F1, achieving the best balance of accuracy, recall, and generalisation. The winning model becomes the deployed classifier (`models/best_model.pkl`).

**Why recall?** False negatives (missed loan acceptors) cost the bank revenue — each missed acceptor is a lost loan. False positives (marketing a non-acceptor) cost only a contact. Optimising for recall minimises the more expensive error.

---

## Strategic Recommendations

### P0 — Launch Income-First Targeting Immediately
Apply the model to the full customer base and prioritise outreach to customers predicted positive with confidence > 0.7. Expected lift: 3–5× response rate over untargeted campaigns.

### P1 — Create a Graduate-Professional Loan Product
Customers with advanced education accept at 3× the rate of undergraduates. Design a loan product with features aligned to this segment (e.g., income growth curves, career transition financing) and market it specifically to this cohort.

### P2 — Bundle CD Account Holders
Customers with CD accounts are highly engaged bank product users and disproportionate loan acceptors. A cross-sell push (CD maturity → personal loan offer) is a low-friction, high-conversion play.

### P3 — Family Size as a Trigger Signal
Families of 3–4 members show elevated loan acceptance. Life events (new child, home purchase, education costs) are natural triggers. Build event-triggered campaign logic tied to family size updates in CRM.

### P4 — Rebuild with Ensemble Models
Decision Trees are interpretable but capped in performance. The next iteration should evaluate Random Forest and XGBoost — likely to push recall above 95% — and add SHAP explanations to maintain interpretability for compliance review.
