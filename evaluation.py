import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from imblearn.metrics import specificity_score
from config import RESULTS_DIR, LOGGER
from utils import find_optimal_threshold


def visualizar_analise_exploratoria_dados(df):
    LOGGER.info("Generating EDA visualizations.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "distribuicao")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='diabetes', data=df, palette='viridis', hue='diabetes', legend=False)
    plt.title('Target Distribution (Diabetes)')
    plt.savefig(os.path.join(save_dir, "dist_target.png"), dpi=300)
    plt.close()


def visualizar_resultados(y_true, y_pred, y_prob, nome_modelo):
    LOGGER.info(f"Generating evaluation plots for {nome_modelo}.")
    display_name = nome_modelo
    try:
        nml = str(nome_modelo).lower()
        if nml.endswith('_train'):
            display_name = str(nome_modelo)[:-6]
    except Exception:
        display_name = nome_modelo
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([["vn", "fp"], ["fn", "vp"]])
    annot = [[f"{labels[i, j]}\n{cm[i, j]}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão - {display_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    ax.set_xticklabels(['Negativo', 'Positivo'])
    ax.set_yticklabels(['Negativo', 'Positivo'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "confusao", f"cm_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve - {display_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "roc", f"roc_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'AP = {ap:.2f}')
    base = np.mean(y_true)
    plt.hlines(base, 0, 1, colors='red', linestyles='--', label=f'Baseline = {base:.2f}')
    plt.title(f'Precision-Recall Curve - {display_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "pr", f"pr_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.title(f'Calibration Curve - {display_name}')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend(loc='best')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "calibracao", f"cal_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def avaliar_modelo(model, x_test, y_test, nome_modelo, is_keras_model=False,
                   threshold_mode: str = 'auto',
                   objective: str = 'f1',
                   min_recall: float | None = 0.6,
                   min_precision: float | None = None,
                   fixed_threshold: float | None = None):
    LOGGER.info(f"Evaluating model: {nome_modelo}")
    if is_keras_model:
        if "cnn" in nome_modelo.lower() or "hibrido" in nome_modelo.lower():
            if len(x_test.shape) == 2:
                x_test = np.expand_dims(x_test, axis=-1)
        y_prob = model.predict(x_test).flatten()
        y_pred_default = (y_prob > 0.5).astype(int)
    else:
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred_default = model.predict(x_test)

    if threshold_mode == 'fixed' and fixed_threshold is not None:
        thr = float(fixed_threshold)
        y_pred = (y_prob > thr).astype(int)
        # Compute stats at fixed threshold
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1v = f1_score(y_test, y_pred, zero_division=0)
        thr_stats = {'precision': p, 'recall': r, 'f1': f1v}
        thr_label = f"fixed@{thr:.3f}"
    else:
        thr, thr_stats = find_optimal_threshold(
            y_test, y_prob,
            objective=objective,
            min_recall=min_recall,
            min_precision=min_precision
        )
        y_pred = (y_prob > thr).astype(int)
        # Build label
        parts = [objective]
        if min_recall is not None:
            parts.append(f"recall>={min_recall:.2f}")
        if min_precision is not None:
            parts.append(f"precision>={min_precision:.2f}")
        thr_label = " ".join(parts)

    confusion_matrix_values = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix_values.ravel()

    specificity = specificity_score(y_test, y_pred) if (tn + fp) > 0 else 0.0
    recall_pos = recall_score(y_test, y_pred, zero_division=0)
    ap = average_precision_score(y_test, y_prob)
    precision_val = precision_score(y_test, y_pred, zero_division=0)

    metrics = {
        'modelo': nome_modelo,
        'threshold_used': thr,
        'threshold_objective': thr_label,
        'thr_precision': thr_stats.get('precision'),
        'thr_recall': thr_stats.get('recall'),
        'thr_f1': thr_stats.get('f1'),
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': ap,
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_val,
        'recall': recall_pos,
        'specificity': specificity,
        'precision_default': precision_score(y_test, y_pred_default, zero_division=0),
        'recall_default': recall_score(y_test, y_pred_default, zero_division=0),
        'f1_default': f1_score(y_test, y_pred_default, zero_division=0),
    }

    pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, f"{nome_modelo.replace(' ', '_').lower()}_metricas.csv"), index=False)

    visualizar_resultados(y_test, y_pred, y_prob, nome_modelo)

    import gerar_graficos as gg
    gg.exportar_metricas_principais(y_test, y_pred, nome_modelo.replace(' ', '_').lower(), metrics)
    gg.exportar_metricas_adicionais(y_test, y_pred, nome_modelo.replace(' ', '_').lower(), metrics)
    return metrics


def plot_metric_comparison(df_metrics, metric, title=None, xlabel=None, filename=None):
    """
    Generate and save a barplot comparing models for a given metric.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame containing model metrics (one row per model).
    metric : str
        Name of the metric column to plot.
    title : Optional[str]
        Title for the plot. If None, uses metric name.
    xlabel : Optional[str]
        X-axis label. If None, uses metric name.
    filename : Optional[str]
        Output filename (PNG). If None, uses metric name.
    """
    df_sorted = df_metrics.sort_values(metric, ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=metric, y='modelo', data=df_sorted, palette='viridis', hue='modelo', dodge=False)
    plt.title(title or f'Model Comparison - {metric}', fontsize=16)
    plt.xlabel(xlabel or metric, fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    fname = filename or f"comparacao_modelos_{metric}.png"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300)
    plt.close()


def comparar_metricas_multiplas(df_metrics, metricas=None):
    """
    Generate and save comparison barplots for multiple metrics across models.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame with one row per model and columns for each metric.
    metricas : Optional[List[str]]
        List of metric names to plot. If None, uses a default set.
    """
    if metricas is None:
        metricas = [
            'accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall',
            'specificity', 'mcc', 'gmean', 'brier', 'log_loss', 'f2', 'f0_5', 'ece', 'mce'
        ]
    for metrica in metricas:
        if metrica in df_metrics.columns:
            plot_metric_comparison(
                df_metrics,
                metric=metrica,
                title=f'Model Comparison - {metrica.replace("_", " ").upper()}',
                xlabel=metrica.replace('_', ' ').title(),
                filename=f"graficos/metricas/comparacao_modelos_{metrica}.png"
            )


def comparar_todos_modelos(df_metrics):
    LOGGER.info("Generating model comparison plot.")
    df_metrics = df_metrics.sort_values('roc_auc', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='roc_auc', y='modelo', data=df_metrics, palette='viridis', hue='modelo', dodge=False)
    plt.title('Model Comparison - ROC AUC', fontsize=16)
    plt.xlabel('ROC AUC Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparacao_modelos_roc_auc.png"), dpi=300)
    plt.close()
    comparar_metricas_multiplas(df_metrics)
