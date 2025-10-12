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
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import geometric_mean_score, specificity_score
from scipy.stats import ks_2samp
from config import RESULTS_DIR, LOGGER


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
    confusion_matrix_values = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {nome_modelo}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "confusao", f"cm_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve - {nome_modelo}')
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
    plt.title(f'Precision-Recall Curve - {nome_modelo}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "pr", f"pr_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.title(f'Calibration Curve - {nome_modelo}')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend(loc='best')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "calibracao", f"cal_{nome_modelo.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def avaliar_modelo(model, x_test, y_test, nome_modelo, is_keras_model=False):
    LOGGER.info(f"Evaluating model: {nome_modelo}")
    if is_keras_model:
        if "cnn" in nome_modelo.lower() or "hibrido" in nome_modelo.lower():
            if len(x_test.shape) == 2:
                x_test = np.expand_dims(x_test, axis=-1)
        y_prob = model.predict(x_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
    confusion_matrix_values = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix_values.ravel()
    specificity = specificity_score(y_test, y_pred) if (tn + fp) > 0 else 0.0
    recall_pos = recall_score(y_test, y_pred)
    fpr_val = 1 - specificity
    fnr_val = 1 - recall_pos
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ap = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmean = geometric_mean_score(y_test, y_pred) if (tp + fn) > 0 and (tn + fp) > 0 else 0.0
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob, labels=[0, 1])
    precision_val = precision_score(y_test, y_pred)
    fdr = 1 - precision_val
    _for = 1 - npv
    f2 = fbeta_score(y_test, y_pred, beta=2)
    f0_5 = fbeta_score(y_test, y_pred, beta=0.5)
    positives = y_prob[y_test == 1]
    negatives = y_prob[y_test == 0]
    ks_stat = 0.0
    if len(positives) > 0 and len(negatives) > 0:
        ks_stat = float(ks_2samp(positives, negatives, alternative='two-sided', mode='auto').statistic)

    def ece_mce_calibracao(y_true, y_scores, n_bins=10):
        y_scores = np.asarray(y_scores)
        y_true = np.asarray(y_true)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(y_scores, bins) - 1
        n = len(y_true)
        ece = 0.0
        mce = 0.0
        for b in range(n_bins):
            mask = idx == b
            if np.any(mask):
                conf = float(np.mean(y_scores[mask]))
                acc = float(np.mean(y_true[mask]))
                gap = abs(acc - conf)
                ece += (np.sum(mask) / n) * gap
                mce = max(mce, gap)
        eps = 1e-6
        p = np.clip(y_scores, eps, 1 - eps)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        try:
            lr = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e6)
            lr.fit(z, y_true)
            calib_intercept = float(lr.intercept_[0])
            calib_slope = float(lr.coef_[0][0])
        except Exception:
            calib_intercept, calib_slope = np.nan, np.nan
        return ece, mce, calib_intercept, calib_slope

    ece, mce, calib_intercept, calib_slope = ece_mce_calibracao(y_test, y_prob, n_bins=10)

    metrics = {
        'modelo': nome_modelo,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': ap,
        'f1': f1_score(y_test, y_pred),
        'precision': precision_val,
        'recall': recall_pos,
        'specificity': specificity,
        'fpr': fpr_val,
        'fnr': fnr_val,
        'npv': npv,
        'mcc': mcc,
        'gmean': gmean,
        'brier': brier,
        'log_loss': ll,
        'f2': f2,
        'f0_5': f0_5,
        'ks': ks_stat,
        'ece': ece,
        'mce': mce,
        'calib_intercept': calib_intercept,
        'calib_slope': calib_slope,
        'fdr': fdr,
        'for': _for,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, f"{nome_modelo.replace(' ', '_').lower()}_metricas.csv"), index=False)
    visualizar_resultados(y_test, y_pred, y_prob, nome_modelo)
    import gerar_graficos as gg
    gg.exportar_metricas_principais(y_test, y_pred, nome_modelo.replace(' ', '_').lower(), metrics)
    gg.exportar_metricas_adicionais(y_test, y_pred, nome_modelo.replace(' ', '_').lower(), metrics)
    return metrics


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