import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def train_default_dt(X_train, y_train) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(random_state=1)
    model.fit(X_train, y_train)
    return model


def train_prepruned_dt(X_train, y_train) -> DecisionTreeClassifier:
    param_grid = {
        "max_depth": list(range(6, 15)),
        "min_samples_leaf": [1, 2, 5, 7],
        "max_leaf_nodes": [2, 3, 5],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=1),
        param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_postpruned_dt(X_train, y_train, X_test, y_test) -> DecisionTreeClassifier:
    path = DecisionTreeClassifier(random_state=1).cost_complexity_pruning_path(X_train, y_train)
    alphas = abs(path.ccp_alphas[:-1])  # exclude trivial last alpha (single leaf)

    best_alpha, best_f1 = 0.0, 0.0
    for alpha in alphas:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=1)
        clf.fit(X_train, y_train)
        score = f1_score(y_test, clf.predict(X_test))
        if score > best_f1:
            best_f1, best_alpha = score, alpha

    best_model = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=1)
    best_model.fit(X_train, y_train)
    return best_model


def save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
