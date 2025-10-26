import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    average_precision_score,
)
from config import RESULTS_DIR, LOGGER


def _display_title_name(nome_modelo: str) -> str:
    s = str(nome_modelo)
    if s.lower().endswith('_train'):
        s = s[:-6]
    if s.lower().endswith('_bayesian'):
        s = s[:-9]
    return s


def visualizar_analise_exploratoria_dados(df):
    LOGGER.info("Generating EDA visualizations.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "distribuicao")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='diabetes', data=df, palette='viridis', hue='diabetes', legend=False)
    plt.title('Target Distribution (Diabetes)')
    plt.savefig(os.path.join(save_dir, "dist_target.png"), dpi=300)
    plt.close()


def visualizar_resultados(y_true, y_pred, nome_modelo):
    LOGGER.info(f"Generating confusion matrix for {nome_modelo}.")
    display_name = _display_title_name(nome_modelo)
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
    save_dir = os.path.join(RESULTS_DIR, "graficos", "confusao")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"cm_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - {_display_title_name(model_name)}')
    plt.legend(loc="lower right")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "roc")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"roc_{model_name.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def plot_pr_curve(y_true, y_score, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'AP = {ap:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall - {_display_title_name(model_name)}')
    plt.legend(loc="lower left")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "pr")
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pr_{model_name.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def avaliar_modelo(model, x_test, y_test, nome_modelo, is_keras_model=False):
    LOGGER.info(f"Evaluating model: {nome_modelo}")
    if is_keras_model:
        if ("cnn" in nome_modelo.lower() or "hibrido" in nome_modelo.lower()) and len(x_test.shape) == 2:
            x_test = np.expand_dims(x_test, axis=-1)
        y_prob = model.predict(x_test).flatten()
    else:
        y_prob = model.predict_proba(x_test)[:, 1]

    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    try:
        loss = float(log_loss(y_test, y_prob, labels=[0, 1]))
    except Exception:
        loss = float('nan')
    try:
        roc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc = float('nan')
    try:
        pr = float(average_precision_score(y_test, y_prob))
    except Exception:
        pr = float('nan')

    metrics = {
        'modelo': nome_modelo,
        'f1': f1,
        'recall': rec,
        'precision': prec,
        'loss': loss,
        'accuracy': acc,
        'roc_auc': roc,
        'pr_auc': pr,
    }

    out_csv = os.path.join(RESULTS_DIR, f"{nome_modelo.replace(' ', '_').lower()}_metricas.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)

    visualizar_resultados(y_test, y_pred, nome_modelo)
    try:
        plot_roc_curve(y_test, y_prob, nome_modelo)
    except Exception as e:
        LOGGER.warning(f"ROC curve failed for {nome_modelo}: {e}")
    try:
        plot_pr_curve(y_test, y_prob, nome_modelo)
    except Exception as e:
        LOGGER.warning(f"PR curve failed for {nome_modelo}: {e}")
    return metrics


def comparar_todos_modelos(df_metrics):
    LOGGER.info("Generating model comparison plot (F1).")
    if 'f1' not in df_metrics.columns:
        LOGGER.warning("F1 column not found in df_metrics; skipping comparison plot.")
        return
    df_metrics = df_metrics.sort_values('f1', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='f1', y='modelo', data=df_metrics, palette='viridis', hue='modelo', dodge=False)
    plt.title('Model Comparison - F1 Score', fontsize=16)
    plt.xlabel('F1 Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparacao_modelos_f1.png"), dpi=300)
    plt.close()
