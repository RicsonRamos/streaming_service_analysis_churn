"""
Evaluation Utilities
Handles metric calculation, threshold optimization,
business metrics, and markdown reporting.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

"""
Evaluation Utilities
Handles metric calculation, threshold optimization,
business metrics, and markdown reporting.
"""


class ModelEvaluator:
    """
    Model Evaluator class.

    Handles metric calculation, threshold optimization,
    business metrics, and markdown reporting.
    """

    def __init__(self, artifacts_path: str, figures_path: str, reports_path: str):
        """
        Initialize the Model Evaluator.

        :param artifacts_path: Path to the artifacts directory.
        :param figures_path: Path to the figures directory.
        :param reports_path: Path to the reports directory.
        """
        self.artifacts_path = Path(artifacts_path)
        self.figures_path = Path(figures_path)
        self.reports_path = Path(reports_path)

        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # Core Metrics
    # ---------------------------------------------------

    def _compute_core_metrics(self, y_true, y_pred, probs):
        """
        Compute the core metrics (ROC-AUC, PR-AUC, accuracy, precision, recall, F1-score, Brier score).

        :param y_true: Actual labels.
        :param y_pred: Predicted labels.
        :param probs: Predicted probabilities.

        :return: A dictionary with the core metrics.
        """
        metrics = {}

        metrics["roc_auc"] = roc_auc_score(y_true, probs)
        metrics["prauc"] = average_precision_score(y_true, probs)

        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred)

        metrics["brier_score"] = brier_score_loss(y_true, probs)

        return metrics

    # ---------------------------------------------------
    # Threshold Optimization
    # ---------------------------------------------------

    def _optimize_threshold(self, y_true, probs):
        """
        Optimize the threshold for the F1-score.

        :param y_true: Actual labels.
        :param probs: Predicted probabilities.

        :return: A dictionary with the best threshold and the corresponding F1-score.
        """
        thresholds = np.linspace(0.05, 0.95, 181)

        best_f1 = 0
        best_t = 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_true, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return {"best_threshold": best_t, "best_f1": best_f1}

    # ---------------------------------------------------
    # Lift Metrics
    # ---------------------------------------------------

    def _compute_lift(self, y_true, probs, top_pct=0.1):
        """
        Compute the lift metrics (lift, churn rate).

        :param y_true: Actual labels.
        :param probs: Predicted probabilities.
        :param top_pct: Top percentage of the sorted data.

        :return: A dictionary with the lift metrics.
        """
        data = np.column_stack([y_true, probs])
        data = data[data[:, 1].argsort()[::-1]]

        n_top = int(len(data) * top_pct)

        top = data[:n_top]

        baseline_rate = y_true.mean()
        top_rate = top[:, 0].mean()

        lift = top_rate / baseline_rate if baseline_rate > 0 else 0

        return {
            f"lift_at_{int(top_pct*100)}pct": lift,
            f"top_{int(top_pct*100)}pct_churn_rate": top_rate,
        }

    # ---------------------------------------------------
    # Public Interface
    # ---------------------------------------------------

    def evaluate(self, y_test, y_pred, probs, model_name: str):
        """
        Evaluate the model.

        :param y_test: Actual labels.
        :param y_pred: Predicted labels.
        :param probs: Predicted probabilities.
        :param model_name: Model name.

        :return: A dictionary with the evaluation metrics.
        """
        metrics = {}

        # Core
        core = self._compute_core_metrics(y_test, y_pred, probs)
        metrics.update(core)

        # Threshold
        threshold_metrics = self._optimize_threshold(y_test, probs)
        metrics.update(threshold_metrics)

        # Lift
        lift_10 = self._compute_lift(y_test, probs, 0.10)
        lift_20 = self._compute_lift(y_test, probs, 0.20)

        metrics.update(lift_10)
        metrics.update(lift_20)

        # Save artifacts
        self._save_json(metrics, model_name)
        self._save_markdown(metrics, model_name)

        # Plots
        self._save_plots(y_test, y_pred, probs, model_name)

        return metrics

    # ---------------------------------------------------
    # Saving
    # ---------------------------------------------------

    def _save_json(self, metrics, model_name):
        """
        Save the metrics to a JSON file.

        :param metrics: Evaluation metrics.
        :param model_name: Model name.
        """
        path = self.artifacts_path / f"{model_name}_metrics.json"

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

    def _save_markdown(self, metrics, model_name):
        """
        Save the metrics to a Markdown file.

        :param metrics: Evaluation metrics.
        :param model_name: Model name.
        """
        path = self.reports_path / f"{model_name}_metrics.md"

        lines = []

        lines.append(f"# Model Evaluation Report — {model_name}\n")

        lines.append("## Core Metrics\n")

        lines.append(f"- ROC-AUC: {metrics['roc_auc']:.4f}")
        lines.append(f"- PR-AUC: {metrics['prauc']:.4f}")
        lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
        lines.append(f"- Precision: {metrics['precision']:.4f}")
        lines.append(f"- Recall: {metrics['recall']:.4f}")
        lines.append(f"- F1-Score: {metrics['f1']:.4f}")
        lines.append(f"- Brier Score: {metrics['brier_score']:.4f}\n")

        lines.append("## Threshold Optimization\n")

        lines.append(f"- Best Threshold (F1): {metrics['best_threshold']:.3f}")
        lines.append(f"- Best F1: {metrics['best_f1']:.4f}\n")

        lines.append("## Business Metrics (Lift)\n")

        lines.append(f"- Lift @10%: {metrics['lift_at_10pct']:.2f}")
        lines.append(f"- Churn Rate @10%: {metrics['top_10pct_churn_rate']:.2f}")

        lines.append(f"- Lift @20%: {metrics['lift_at_20pct']:.2f}")
        lines.append(f"- Churn Rate @20%: {metrics['top_20pct_churn_rate']:.2f}\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ---------------------------------------------------
    # Plots
    # ---------------------------------------------------

    def _save_plots(self, y_true, y_pred, probs, model_name):
        """
        Save the plots to files.

        :param y_true: Actual labels.
        :param y_pred: Predicted labels.
        :param probs: Predicted probabilities.
        :param model_name: Model name.
        """
        # Confusion Matrix
        plt.figure(figsize=(6, 5))

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(f"Confusion Matrix — {model_name}")

        plt.savefig(self.figures_path / f"{model_name}_confusion_matrix.png")

        plt.close()

        # ROC
        fpr, tpr, _ = roc_curve(y_true, probs)

        plt.figure(figsize=(6, 5))

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probs):.3f}")
        plt.plot([0, 1], [0, 1], "--")

        plt.legend()

        plt.title(f"ROC Curve — {model_name}")

        plt.savefig(self.figures_path / f"{model_name}_roc_curve.png")

        plt.close()
