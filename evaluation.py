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

from config import RESULTS_DIR, LOGGER


def visualizar_analise_exploratoria_dados(df):
    """Cria visualizações para a análise exploratória dos dados."""
    LOGGER.info("Gerando visualizações para análise exploratória.")
    save_dir = os.path.join(RESULTS_DIR, "graficos", "distribuicao")

    # Distribuição da variável alvo
    plt.figure(figsize=(8, 6))
    sns.countplot(x='diabetes', data=df, palette='viridis', hue='diabetes', legend=False)
    plt.title('Distribuição da Variável Alvo (Diabetes)')
    plt.savefig(os.path.join(save_dir, "dist_target.png"), dpi=300)
    plt.close()

    # Outras visualizações...
    # (Adicione aqui o restante do código da sua função original 'visualizar_analise_exploratoria')


def visualizar_resultados(y_true, y_pred, y_prob, nome_modelo):
    """Gera e salva gráficos de avaliação para um modelo."""
    LOGGER.info(f"Gerando gráficos de avaliação para {nome_modelo}.")

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "confusao", f"cm_{nome_modelo.replace(' ', '_').lower()}.png"),
                dpi=300)
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'Curva ROC - {nome_modelo}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "roc", f"roc_{nome_modelo.replace(' ', '_').lower()}.png"),
                dpi=300)
    plt.close()

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}')
    base = np.mean(y_true)
    plt.hlines(base, 0, 1, colors='red', linestyles='--', label=f'Base = {base:.2f}')
    plt.title(f'Curva Precision-Recall - {nome_modelo}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "pr", f"pr_{nome_modelo.replace(' ', '_').lower()}.png"),
                dpi=300)
    plt.close()

    # Curva de Calibração
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfeito')
    plt.title(f'Curva de Calibração - {nome_modelo}')
    plt.xlabel('Probabilidade prevista média')
    plt.ylabel('Frequência positiva')
    plt.legend(loc='best')
    plt.savefig(os.path.join(RESULTS_DIR, "graficos", "calibracao", f"cal_{nome_modelo.replace(' ', '_').lower()}.png"),
                dpi=300)
    plt.close()


def avaliar_modelo(model, x_test, y_test, nome_modelo, is_keras_model=False):
    """Avalia um modelo e retorna um dicionário de métricas."""
    LOGGER.info(f"Avaliando modelo: {nome_modelo}")

    if is_keras_model:
        if "cnn" in nome_modelo.lower() or "hibrido" in nome_modelo.lower():
            if len(x_test.shape) == 2:
                x_test = np.expand_dims(x_test, axis=-1)
        y_prob = model.predict(x_test).flatten()
        y_pred = (y_prob > 0.5).astype(int)
    else:  # Modelo Sklearn
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ap = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmean = np.sqrt(recall_score(y_test, y_pred) * specificity)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob, labels=[0, 1])
    metrics = {
        'modelo': nome_modelo,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': ap,
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'specificity': specificity,
        'fpr': fpr_val,
        'fnr': fnr_val,
        'npv': npv,
        'mcc': mcc,
        'gmean': gmean,
        'brier': brier,
        'log_loss': ll,
    }

    # Salvar e visualizar resultados
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, f"{nome_modelo.replace(' ', '_').lower()}_metricas.csv"),
                                   index=False)
    visualizar_resultados(y_test, y_pred, y_prob, nome_modelo)
    from gerar_graficos import export_all_metrics
    export_all_metrics(y_test, y_pred, nome_modelo.replace(' ', '_').lower(), metrics)

    return metrics


def comparar_todos_modelos(df_metrics):
    """Cria um gráfico comparando as métricas de todos os modelos."""
    LOGGER.info("Gerando gráfico de comparação de modelos.")
    df_metrics = df_metrics.sort_values('roc_auc', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='roc_auc', y='modelo', data=df_metrics, palette='viridis', hue='modelo', dodge=False)
    plt.title('Comparação de Modelos - ROC AUC', fontsize=16)
    plt.xlabel('ROC AUC Score', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparacao_modelos_roc_auc.png"), dpi=300)
    plt.close()