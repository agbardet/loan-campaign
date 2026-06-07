import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix,
)


def classification_report_dict(model, X, y, label: str) -> dict:
    y_pred = model.predict(X)
    return {
        "model": label,
        "accuracy": round(accuracy_score(y, y_pred), 3),
        "recall": round(recall_score(y, y_pred), 3),
        "precision": round(precision_score(y, y_pred), 3),
        "f1": round(f1_score(y, y_pred), 3),
    }


def plot_confusion_matrix(model, X_test, y_test, title: str, save_path: str) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = np.array([[f"{v}\n({lbl})" for v, lbl in zip(row, lbl_row)]
                      for row, lbl_row in zip(cm, labels)])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", linewidths=0.5,
                xticklabels=["No Loan", "Loan"], yticklabels=["No Loan", "Loan"], ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, save_path: str) -> None:
    importances = model.feature_importances_
    feat_df = (
        __import__("pandas").DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )
    feat_df = feat_df[feat_df["importance"] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x="importance", y="feature",
                hue="feature", palette="Blues_d", legend=False, ax=ax)
    ax.set_title("Feature Importance — Post-Pruned Decision Tree",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_ylabel("")
    for bar in ax.patches:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_models(results: list, save_path: str) -> None:
    import pandas as pd
    df = pd.DataFrame(results)
    metrics = ["recall", "precision", "f1"]
    df_melted = df.melt(id_vars="model", value_vars=metrics,
                        var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df_melted, x="model", y="score", hue="metric",
                palette={"recall": "#2E86AB", "precision": "#A8DADC", "f1": "#1D3557"}, ax=ax)
    ax.set_title("Model Comparison — Recall, Precision, F1 (Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.7, 1.02)
    ax.legend(title="Metric", fontsize=10)
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
