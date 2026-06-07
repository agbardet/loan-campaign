# Data Dictionary — AllLife Bank Loan Modelling

**Source:** AllLife Bank customer dataset  
**Rows:** 5,000 · **Columns:** 14 (12 features + 1 ID + 1 target)

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `ID` | int | Unique customer identifier | Dropped — not a predictive feature |
| `Age` | int | Customer age in years | Range 23–67 |
| `Experience` | int | Years of professional experience | Dropped — r≈0.99 correlation with Age makes it redundant |
| `Income` | int | Annual income in thousands USD | Strongest predictor of loan acceptance |
| `ZIPCode` | int | Home address ZIP code | Truncated to 2-digit prefix, then one-hot encoded (drop_first=True) |
| `Family` | int | Family size (number of members) | 1–4; second-strongest predictor |
| `CCAvg` | float | Average monthly credit card spend in thousands USD | Proxy for discretionary income |
| `Education` | int | Education level: 1=Undergrad, 2=Graduate, 3=Advanced/Professional | Ordinal; higher education → higher acceptance rate |
| `Mortgage` | int | Value of home mortgage in thousands USD | Many zeros (no mortgage); right-skewed |
| `Personal_Loan` | int | **Target** — did customer accept personal loan offer? 0=No, 1=Yes | 9.6% positive class (480 of 5,000) — imbalanced |
| `Securities_Account` | int | Does customer have a securities account? 0=No, 1=Yes | Binary indicator |
| `CD_Account` | int | Does customer have a certificate of deposit account? 0=No, 1=Yes | Binary indicator |
| `Online` | int | Does customer use internet banking? 0=No, 1=Yes | Binary indicator |
| `CreditCard` | int | Does customer have a bank-issued credit card? 0=No, 1=Yes | Binary indicator |

## Class Imbalance Note

`Personal_Loan` has a 9.6% positive rate (480 yes / 4,520 no). This imbalance means:
- Accuracy is a misleading metric — a model predicting all-No achieves 90.4% accuracy
- Model selection prioritises **Recall** (catching true loan acceptors) and **F1** (balance of precision and recall)
- Train/test split uses `stratify=y` to preserve class ratio in both sets
