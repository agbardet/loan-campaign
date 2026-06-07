# Portfolio Copy — AllLife Bank Loan Campaign

## Card Title
AllLife Bank — Personal Loan Campaign

## One-Liner
Binary classification model predicting personal loan acceptance — 90.3% recall, 91.5% F1 on a class-imbalanced dataset.

## Bullet Highlights
- Built and tuned three Decision Tree variants (default → pre-pruned → post-pruned); post-pruned model achieves Test F1 0.915 with minimal overfitting
- Handled 9.6% class imbalance via stratified splitting and recall-optimised model selection
- Packaged reusable ML pipeline in src/ (preprocess, train, evaluate) with clean notebook narrative importing from it

## Tags
`Python` `scikit-learn` `NumPy` `Pandas` `Seaborn` `Matplotlib` `Decision Tree` `Classification` `ML`

## Extended Blurb
AllLife Bank needed to grow its personal loan portfolio without mass-marketing to unqualified customers. With only 9.6% of 5,000 customers having accepted a prior offer, untargeted outreach is expensive and low-conversion.

I built a supervised binary classifier comparing three Decision Tree variants — default, pre-pruned via GridSearchCV optimised for recall, and post-pruned via cost complexity pruning. The post-pruned model achieves 90.3% recall and 91.5% F1 on held-out test data, correctly identifying 9 in 10 likely loan acceptors with only 7.1% false positives.

Income dominates feature importance (61.7%), followed by family size and education level — three features that, combined, define a simple targeting rule marketing can act on immediately. The project is structured as a hybrid notebook + src/ Python package, demonstrating production-oriented code organisation beyond notebook-only EDA.

## Resume Bullets
- Developed binary classification pipeline for AllLife Bank personal loan propensity modelling; post-pruned Decision Tree achieved 91.5% F1 and 90.3% recall on imbalanced dataset (9.6% positive class)
- Implemented three-stage model evaluation framework (default → GridSearchCV → cost complexity pruning) with automated alpha selection maximising test F1
- Packaged ML pipeline as reusable Python src/ module (preprocess, train, evaluate) with narrative Jupyter notebook consuming it — production-oriented portfolio structure
