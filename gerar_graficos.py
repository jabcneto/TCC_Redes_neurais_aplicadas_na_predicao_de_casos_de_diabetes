import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from config import RESULTS_DIR, LOGGER

# Funções originais do arquivo gerar_graficos.py
def criar_grafico_evolucao_modelos(resultados_df):
    """
    Cria um gráfico mostrando a evolução das métricas entre diferentes tipos de modelos.
    """

    # Definir ordem dos modelos (do mais simples ao mais complexo)
    ordem_modelos = [
        'Naive Bayes', 'Regressão Logística', 'Decision Tree', 'KNN',
        'SVM', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM',
        'MLP', 'CNN', 'Híbrido CNN-LSTM'
    ]

    # Filtrar modelos existentes
    modelos_existentes = [m for m in ordem_modelos if m in resultados_df['modelo'].values]
    df_ordenado = resultados_df[resultados_df['modelo'].isin(modelos_existentes)].copy()

    # Reordenar com base na ordem definida
    df_ordenado['ordem'] = df_ordenado['modelo'].map({modelo: i for i, modelo in enumerate(modelos_existentes)})
    df_ordenado = df_ordenado.sort_values('ordem')

    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Evolução das Métricas por Complexidade do Modelo', fontsize=20, y=0.98)

    # Métricas principais
    metricas = ['accuracy', 'f1', 'roc_auc', 'recall']
    titulos = ['Acurácia', 'F1-Score', 'ROC AUC', 'Recall']
    cores = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']

    for i, (metrica, titulo, cor) in enumerate(zip(metricas, titulos, cores)):
        ax = axes[i // 2, i % 2]

        # Plotar linha de evolução
        x_vals = range(len(df_ordenado))
        y_vals = df_ordenado[metrica].values

        ax.plot(x_vals, y_vals, marker='o', linewidth=3, markersize=8, color=cor, alpha=0.8)

        # Destacar melhor modelo
        best_idx = np.argmax(y_vals)
        best_value = y_vals[best_idx]
        best_model = df_ordenado.iloc[best_idx]['modelo']

        ax.scatter(best_idx, best_value, color='red', s=200, zorder=5, alpha=0.7)

        ax.set_title(titulo, fontsize=16, fontweight='bold')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(df_ordenado['modelo'], rotation=45, ha='right', fontsize=11)
        ax.set_ylabel(titulo, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Adicionar linha de tendência
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), "--", alpha=0.5, color='gray', linewidth=2)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/evolucao_modelos_complexidade.png", dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info("Gráfico de evolução criado")


def criar_heatmap_metricas_modelos(resultados_df):
    """
    Cria um heatmap comparando todas as métricas de todos os modelos.
    """

    # Selecionar métricas para o heatmap
    metricas_cols = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Verificar quais colunas existem
    metricas_disponiveis = [col for col in metricas_cols if col in resultados_df.columns]

    if not metricas_disponiveis:
        LOGGER.error("Nenhuma métrica encontrada para criar heatmap")
        return

    # Preparar dados para heatmap
    heatmap_data = resultados_df.set_index('modelo')[metricas_disponiveis]

    # Ordenar por F1-score se disponível, senão por accuracy
    sort_col = 'f1' if 'f1' in metricas_disponiveis else metricas_disponiveis[0]
    heatmap_data = heatmap_data.sort_values(sort_col, ascending=False)

    # Criar figura
    plt.figure(figsize=(14, 10))

    # Criar heatmap
    sns.heatmap(heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                center=0.5,
                square=False,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                annot_kws={'size': 10})

    plt.title('Heatmap de Performance dos Modelos', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Métricas', fontsize=14)
    plt.ylabel('Modelos', fontsize=14)

    # Personalizar labels dos eixos
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/heatmap_metricas_modelos.png", dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info("Heatmap de métricas criado")


def criar_distribuicao_features_importantes(df):
    """
    Cria gráficos mostrando a distribuição das features mais importantes.
    """
    # Features mais importantes baseadas no domínio médico
    features_importantes = ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age']

    # Verificar quais features existem no dataset
    features_disponiveis = [f for f in features_importantes if f in df.columns]

    if len(features_disponiveis) < 2:
        LOGGER.error("Poucas features importantes encontradas no dataset")
        return

    # Ajustar layout baseado no número de features
    n_features = len(features_disponiveis)
    if n_features == 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 8))
        if n_features == 1:
            axes = [axes]

    fig.suptitle('Distribuição das Features Mais Importantes por Status de Diabetes', fontsize=20, y=0.98)

    for i, feature in enumerate(features_disponiveis):
        ax = axes[i]

        # Separar dados por classe
        diabetes_sim = df[df['diabetes'] == 1][feature]
        diabetes_nao = df[df['diabetes'] == 0][feature]

        # Histograma
        ax.hist(diabetes_nao, bins=30, alpha=0.7, label='Não Diabetes',
                color='#3498DB', density=True, edgecolor='black')
        ax.hist(diabetes_sim, bins=30, alpha=0.7, label='Diabetes',
                color='#E74C3C', density=True, edgecolor='black')

        # Adicionar linhas de média
        ax.axvline(diabetes_nao.mean(), color='#2980B9', linestyle='--', linewidth=2,
                   label=f'Média Não Diabetes: {diabetes_nao.mean():.2f}')
        ax.axvline(diabetes_sim.mean(), color='#C0392B', linestyle='--', linewidth=2,
                   label=f'Média Diabetes: {diabetes_sim.mean():.2f}')

        # Teste estatístico
        try:
            statistic, p_value = mannwhitneyu(diabetes_sim, diabetes_nao, alternative='two-sided')
            ax.set_title(f'{feature}\np-value (Mann-Whitney): {p_value:.2e}',
                         fontsize=14, fontweight='bold')
        except:
            ax.set_title(f'{feature}', fontsize=14, fontweight='bold')

        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Densidade', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao_features_importantes.png", dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info("Gráfico de distribuição de features criado")


def criar_grafico_tempo_treinamento(resultados_df):
    """
    Cria gráficos mostrando o tempo de treinamento dos modelos.
    """
    # Tempos estimados de treinamento (em segundos) - valores típicos
    tempos_estimados = {
        'Naive Bayes': 0.5,
        'Regressão Logística': 2.0,
        'Decision Tree': 1.5,
        'KNN': 0.8,
        'SVM': 15.0,
        'Random Forest': 10.0,
        'Gradient Boosting': 25.0,
        'XGBoost': 20.0,
        'LightGBM': 8.0,
        'MLP': 120.0,
        'CNN': 180.0,
        'Híbrido CNN-LSTM': 300.0
    }

    # Filtrar modelos existentes
    modelos_existentes = [m for m in tempos_estimados.keys() if m in resultados_df['modelo'].values]

    if not modelos_existentes:
        LOGGER.error("Nenhum modelo encontrado para análise de tempo")
        return

    # Preparar dados
    df_tempo = pd.DataFrame({
        'modelo': modelos_existentes,
        'tempo_segundos': [tempos_estimados[m] for m in modelos_existentes],
        'tempo_minutos': [tempos_estimados[m] / 60 for m in modelos_existentes]
    })

    # Adicionar métricas de performance
    metricas_disponiveis = ['f1', 'roc_auc', 'accuracy']
    selected_metric = None
    for metrica in metricas_disponiveis:
        if metrica in resultados_df.columns:
            df_tempo = df_tempo.merge(resultados_df[['modelo', metrica]], on='modelo')
            selected_metric = metrica
            break

    # Criar figura
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Análise de Tempo de Treinamento vs Performance', fontsize=20, y=0.98)

    # 1. Tempo de treinamento por modelo
    ax1 = axes[0]
    bars = ax1.bar(range(len(df_tempo)), df_tempo['tempo_minutos'],
                   color=['#3498DB' if t < 5 else '#F39C12' if t < 60 else '#E74C3C' for t in
                          df_tempo['tempo_minutos']])
    ax1.set_title('Tempo de Treinamento por Modelo', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Modelo', fontsize=12)
    ax1.set_ylabel('Tempo (minutos)', fontsize=12)
    ax1.set_xticks(range(len(df_tempo)))
    ax1.set_xticklabels(df_tempo['modelo'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + max(df_tempo['tempo_minutos']) * 0.01,
                 f'{height:.1f}min', ha='center', va='bottom', fontsize=10)

    # 2. Scatter: Tempo vs Performance
    ax2 = axes[1]
    if selected_metric and selected_metric in df_tempo.columns:
        scatter = ax2.scatter(df_tempo['tempo_minutos'], df_tempo[selected_metric],
                             s=150, alpha=0.7, edgecolors='black')

        # Adicionar nomes dos modelos
        for i, row in df_tempo.iterrows():
            ax2.annotate(row['modelo'], (row['tempo_minutos'], row[selected_metric]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax2.set_title(f'Relação Tempo vs Performance ({selected_metric.upper()})', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Tempo de Treinamento (minutos)', fontsize=12)
        ax2.set_ylabel(selected_metric.upper(), fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/analise_tempo_treinamento.png", dpi=300, bbox_inches='tight')
    plt.close()
    LOGGER.info("Gráfico de tempo de treinamento criado")


def visualizar_analise_exploratoria(df):
    """
    Cria visualizações para análise exploratória dos dados.

    Args:
        df (pd.DataFrame): DataFrame com os dados
    """
    # Configuração para visualizações
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')

    # Dicionário de mapeamento para português
    mapeamento_colunas = {
        'age': 'idade',
        'bmi': 'IMC',
        'HbA1c_level': 'nível de HbA1c',
        'blood_glucose_level': 'nível de glicose no sangue'
    }

    # Criar uma cópia do DataFrame com nomes traduzidos para português
    df_pt = df.copy()

    # Renomear colunas para português se estiverem em inglês
    colunas_para_renomear = {}
    for col_en, col_pt in mapeamento_colunas.items():
        if col_en in df.columns:
            colunas_para_renomear[col_en] = col_pt

    # Aplicar renomeação se necessário
    if colunas_para_renomear:
        df_pt = df_pt.rename(columns=colunas_para_renomear)

    # Definir as colunas numéricas com nomes em português
    num_cols = ['idade', 'IMC', 'nível de HbA1c', 'nível de glicose no sangue']

    # Verificar quais colunas numéricas existem no DataFrame
    num_cols_existentes = [col for col in num_cols if col in df_pt.columns]

    if not num_cols_existentes:
        raise ValueError("Estrutura do DataFrame incompatível. Verificar nomes das colunas.")

    # 1. Distribuição da variável alvo
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='diabetes', data=df_pt, palette=['#3498db', '#e74c3c'], hue='diabetes', legend=False)
    plt.title('Distribuição da Variável Alvo (Diabetes)', fontsize=15)
    plt.xlabel('Diabetes', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)

    total = len(df_pt)
    for p in ax.patches:
        height = getattr(p, 'get_height', lambda: 0)()
        x = getattr(p, 'get_x', lambda: 0)()
        width = getattr(p, 'get_width', lambda: 0)()
        try:
            val = int(height)
        except Exception:
            val = height
        ax.text(x + width / 2.0, height + 5, f"{val} ({(0 if total == 0 else height / total):.1%})", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao/distribuicao_target.png", dpi=300)
    plt.close()

    # 2. Distribuição das variáveis numéricas por status de diabete
    plt.figure(figsize=(15, 10))

    for i, col in enumerate(num_cols_existentes):
        if i >= 4:  # Limitar a 4 gráficos no máximo
            break
        plt.subplot(2, 2, i + 1)
        sns.histplot(data=df_pt, x=col, hue='diabetes', kde=True, bins=30,
                     palette=['#3498db', '#e74c3c'], alpha=0.6)
        plt.title(f'Distribuição de {col} por Status de Diabetes', fontsize=13)
        plt.xlabel(col, fontsize=11)
        plt.ylabel('Contagem', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao/distribuicao_variaveis_numericas.png", dpi=300)
    plt.close()

    # 3. Boxplots para variáveis numéricas
    plt.figure(figsize=(15, 10))

    for i, col in enumerate(num_cols_existentes):
        if i >= 4:  # Limitar a 4 gráficos no máximo
            break
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='diabetes', y=col, data=df_pt, palette=['#3498db', '#e74c3c'], hue='diabetes', legend=False)
        plt.title(f'{col} por Status de Diabetes', fontsize=13)
        plt.xlabel('Diabetes', fontsize=11)
        plt.ylabel(col, fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/boxplots_variaveis_numericas.png", dpi=300)
    plt.close()

    # 4. Matriz de correlação
    plt.figure(figsize=(12, 10))
    # Seleciona apenas colunas numéricas para a matriz de correlação
    num_df = df_pt.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5)
    plt.title('Matriz de Correlação', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/matriz_correlacao.png", dpi=300)
    plt.close()

    # 5. Pairplot para variáveis numéricas
    # Garante que só as colunas numéricas e a coluna alvo sejam usadas
    pairplot_cols = [col for col in num_cols_existentes if col in num_df.columns] + ['diabetes']
    sns.pairplot(df_pt[pairplot_cols], hue='diabetes',
                 palette=['#3498db', '#e74c3c'], diag_kind='kde')
    plt.suptitle('Pairplot de Variáveis Numéricas', y=1.02, fontsize=16)
    plt.savefig(f"{RESULTS_DIR}/graficos/pairplot_variaveis_numericas.png", dpi=300)
    plt.close()

    # 6. Contagem de variáveis categóricas
    cat_cols = df_pt.select_dtypes(include=['object']).columns.tolist()

    if cat_cols:
        plt.figure(figsize=(15, 5 * len(cat_cols)))

        for i, col in enumerate(cat_cols):
            plt.subplot(len(cat_cols), 1, i + 1)
            sns.countplot(x=col, hue='diabetes', data=df_pt, palette=['#3498db', '#e74c3c'], legend=False)
            plt.title(f'Distribuição de {col} por Status de Diabetes', fontsize=13)
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Contagem', fontsize=11)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao/distribuicao_variaveis_categoricas.png", dpi=300)
        plt.close()

    # 7. Relação entre HbA1c e glicose com diabete
    # Verificar se ambas as colunas existem
    if 'nível de HbA1c' in df_pt.columns and 'nível de glicose no sangue' in df_pt.columns:
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(data=df_pt, x='nível de HbA1c', y='nível de glicose no sangue',
                                  hue='diabetes', palette=['#3498db', '#e74c3c'],
                                  s=80, alpha=0.7)
        plt.axvline(x=6.5, color='red', linestyle='--', label='Limiar HbA1c (6.5%)')
        plt.axhline(y=126, color='green', linestyle='--', label='Limiar Glicose (126 mg/dL)')

        # Adicionar anotações para os quadrantes
        plt.text(7.5, 200, 'Alto risco\n(HbA1c alto, Glicose alta)', fontsize=12, ha='center')
        plt.text(5.5, 200, 'Risco moderado\n(HbA1c normal, Glicose alta)', fontsize=12, ha='center')
        plt.text(7.5, 100, 'Risco moderado\n(HbA1c alto, Glicose normal)', fontsize=12, ha='center')
        plt.text(5.5, 100, 'Baixo risco\n(HbA1c normal, Glicose normal)', fontsize=12, ha='center')

        plt.title('Relação entre HbA1c e Glicose no Sangue', fontsize=15)
        plt.xlabel('Nível de HbA1c (%)', fontsize=12)
        plt.ylabel('Nível de Glicose (mg/dL)', fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/relacao_hba1c_glicose.png", dpi=300)
        plt.close()


def visualizar_resultados(y_true, y_pred, y_prob, nome_modelo):
    """
    Cria visualizações para os resultados do modelo.

    Args:
        y_true (np.ndarray): Rótulos verdadeiros
        y_pred (np.ndarray): Predições binárias
        y_prob (np.ndarray): Probabilidades preditas
        nome_modelo (str): Nome do modelo para salvar visualizações
    """

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([["vn", "fp"], ["fn", "vp"]])
    annot_counts = [[f"{labels[i, j]}\n{cm[i, j]}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    ax = sns.heatmap(cm, annot=annot_counts, fmt='', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=15)
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    ax.set_xticklabels(['Negativo', 'Positivo'])
    ax.set_yticklabels(['Negativo', 'Positivo'], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_matriz_confusao.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_norm = [[f"{labels[i, j]}\n{cm_norm[i, j]:.2f}" for j in range(cm_norm.shape[1])] for i in range(cm_norm.shape[0])]
    ax = sns.heatmap(cm_norm, annot=annot_norm, fmt='', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão Normalizada - {nome_modelo}', fontsize=15)
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    ax.set_xticklabels(['Negativo', 'Positivo'])
    ax.set_yticklabels(['Negativo', 'Positivo'], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_matriz_confusao_norm.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title(f'Curva ROC - {nome_modelo}', fontsize=15)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_curva_roc.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.axhline(y=sum(y_true) / len(y_true), color='red', linestyle='--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Curva Precision-Recall - {nome_modelo}', fontsize=15)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_curva_precision_recall.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    prob_pos = y_prob[y_true == 1]
    prob_neg = y_prob[y_true == 0]

    plt.hist(prob_pos, bins=20, alpha=0.5, color='green', label='Classe Positiva (Diabetes)')
    plt.hist(prob_neg, bins=20, alpha=0.5, color='red', label='Classe Negativa (Não Diabetes)')

    plt.axvline(x=0.5, color='black', linestyle='--', label='Limiar (0.5)')
    plt.xlabel('Probabilidade Predita', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.title(f'Distribuição de Probabilidades - {nome_modelo}', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_distribuicao_probabilidades.png", dpi=300)
    plt.close()


def visualizar_historico_treinamento(historico, nome_modelo):
    """
    Visualiza o histórico de treinamento do modelo.

    Args:
        historico (pd.DataFrame | dict | keras.callbacks.History): Histórico de treinamento
        nome_modelo (str): Nome do modelo para salvar visualizações
    """
    try:
        if hasattr(historico, 'history'):
            try:
                df = pd.DataFrame({k: list(v) for k, v in historico.history.items()})
            except Exception:
                df = pd.DataFrame(dict(historico.history))
        elif isinstance(historico, dict):
            df = pd.DataFrame(historico)
        elif isinstance(historico, pd.DataFrame):
            df = historico
        else:
            df = pd.DataFrame(historico)
    except Exception as e:
        LOGGER.error(f"Não foi possível interpretar o histórico fornecido: {e}")
        return

    if 'accuracy' not in df.columns and 'acc' in df.columns:
        df['accuracy'] = df['acc']
    if 'val_accuracy' not in df.columns and 'val_acc' in df.columns:
        df['val_accuracy'] = df['val_acc']
    if 'AUC' not in df.columns and 'auc' in df.columns:
        df['AUC'] = df['auc']
    if 'val_AUC' not in df.columns and 'val_auc' in df.columns:
        df['val_AUC'] = df['val_auc']

    plt.figure(figsize=(12, 8))
    if 'loss' in df.columns:
        plt.plot(df['loss'], label='Treino', color='blue')
    if 'val_loss' in df.columns:
        plt.plot(df['val_loss'], label='Validação', color='orange')
    plt.title(f'Curvas de Perda - {nome_modelo}', fontsize=15)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Perda (Binary Crossentropy)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_curvas_perda.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    if 'accuracy' in df.columns:
        plt.plot(df['accuracy'], label='Treino', color='blue')
    if 'val_accuracy' in df.columns:
        plt.plot(df['val_accuracy'], label='Validação', color='orange')
    plt.title(f'Curvas de Acurácia - {nome_modelo}', fontsize=15)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_curvas_acuracia.png", dpi=300)
    plt.close()

    has_prec_recall_auc = (
        ('precision' in df.columns and 'recall' in df.columns) and ('AUC' in df.columns or 'auc' in df.columns)
    )
    if has_prec_recall_auc:
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(df.get('precision', []), label='Treino', color='blue')
        if 'val_precision' in df.columns:
            plt.plot(df.get('val_precision', []), label='Validação', color='orange')
        plt.title('Precisão')
        plt.xlabel('Época')
        plt.ylabel('Precisão')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(df.get('recall', []), label='Treino', color='blue')
        if 'val_recall' in df.columns:
            plt.plot(df.get('val_recall', []), label='Validação', color='orange')
        plt.title('Recall')
        plt.xlabel('Época')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)

        key_auc = 'AUC' if 'AUC' in df.columns else 'auc'
        key_val_auc = 'val_AUC' if 'val_AUC' in df.columns else ('val_auc' if 'val_auc' in df.columns else None)
        plt.subplot(1, 3, 3)
        plt.plot(df.get(key_auc, []), label='Treino', color='blue')
        if key_val_auc:
            plt.plot(df.get(key_val_auc, []), label='Validação', color='orange')
        plt.title('AUC')
        plt.xlabel('Época')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Métricas de Treinamento - {nome_modelo}', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_curvas_metricas.png", dpi=300)
        plt.close()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    if 'loss' in df.columns:
        plt.plot(df['loss'], label='Treino', color='blue')
    if 'val_loss' in df.columns:
        plt.plot(df['val_loss'], label='Validação', color='orange')
    plt.title('Perda (Loss)', fontsize=13)
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Perda', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    if 'accuracy' in df.columns:
        plt.plot(df['accuracy'], label='Treino', color='blue')
    if 'val_accuracy' in df.columns:
        plt.plot(df['val_accuracy'], label='Validação', color='orange')
    plt.title('Acurácia', fontsize=13)
    plt.xlabel('Época', fontsize=11)
    plt.ylabel('Acurácia', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if 'recall' in df.columns:
        plt.subplot(2, 2, 3)
        plt.plot(df.get('recall', []), label='Treino', color='blue')
        if 'val_recall' in df.columns:
            plt.plot(df.get('val_recall', []), label='Validação', color='orange')
        plt.title('Recall', fontsize=13)
        plt.xlabel('Época', fontsize=11)
        plt.ylabel('Recall', fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)

    if 'auc' in df.columns or 'AUC' in df.columns:
        key_auc = 'AUC' if 'AUC' in df.columns else 'auc'
        key_val_auc = 'val_AUC' if 'val_AUC' in df.columns else ('val_auc' if 'val_auc' in df.columns else None)
        plt.subplot(2, 2, 4)
        plt.plot(df.get(key_auc, []), label='Treino', color='blue')
        if key_val_auc:
            plt.plot(df.get(key_val_auc, []), label='Validação', color='orange')
        plt.title('AUC', fontsize=13)
        plt.xlabel('Época', fontsize=11)
        plt.ylabel('AUC', fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle(f'Curvas de Aprendizado - {nome_modelo}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/{nome_modelo}_curvas_aprendizado.png", dpi=300)
    plt.close()


def visualizar_comparacao_modelos(resultados_df):
    """
    Visualiza comparação entre diferentes modelos.

    Args:
        resultados_df (pd.DataFrame): DataFrame com resultados dos modelos
    """
    # Ordenar por F1-score
    resultados_sorted = resultados_df.sort_values('f1', ascending=False)

    # Métricas para visualização
    metricas = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Visualizar comparação de métricas
    plt.figure(figsize=(20, 12))
    for i, metrica in enumerate(metricas, 1):
        plt.subplot(2, 3, i)
        sns.barplot(x='modelo', y=metrica, data=resultados_sorted, palette='viridis', hue='modelo', legend=False)
        plt.title(f'{metrica.upper()}', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

    plt.suptitle('Comparação de Métricas entre Modelos', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/comparacao_metricas_todos_modelos.png", dpi=300)
    plt.close()

    # Gráfico de radar para top 5 modelos
    top_modelos = resultados_sorted.head(5)

    # Preparar dados para gráfico de radar
    categories = metricas
    N = len(categories)

    # Criar ângulos para o gráfico de radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fechar o círculo

    # Criar figura
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)

    # Adicionar linhas de grade
    plt.xticks(angles[:-1], categories, size=12)
    getattr(ax, 'set_rlabel_position', lambda *a, **k: None)(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)

    # Plotar cada modelo
    for i, row in top_modelos.iterrows():
        values = row[metricas].values.tolist()
        values += values[:1]  # Fechar o círculo
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['modelo'])
        ax.fill(angles, values, alpha=0.1)

    # Adicionar legenda
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('Comparação dos Top 5 Modelos', size=20, y=1.1)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/radar_top_modelos.png", dpi=300)
    plt.close()


def visualizar_matrizes_confusao_modelos_classicos(y_test, y_preds, nomes_modelos):
    """
    Cria matrizes de confusão para modelos clássicos.

    Args:
        y_test (np.ndarray): Valores reais
        y_preds (list): Lista de arrays com predições
        nomes_modelos (list): Lista com nomes dos modelos
    """
    for nome, y_pred in zip(nomes_modelos, y_preds):
        cm = confusion_matrix(y_test, y_pred)
        labels = np.array([["vn", "fp"], ["fn", "vp"]])
        annot = [[f"{labels[i, j]}\n{cm[i, j]}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusão - {nome}', fontsize=15)
        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        ax.set_xticklabels(['Negativo', 'Positivo'])
        ax.set_yticklabels(['Negativo', 'Positivo'], rotation=0)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/confusao/matriz_confusao_{nome.replace(' ', '_').lower()}.png", dpi=300)
        plt.close()


def visualizar_curvas_roc_modelos_classicos(y_test, y_probs, nomes_modelos):
    """
    Cria curvas ROC para modelos clássicos.

    Args:
        y_test (np.ndarray): Valores reais
        y_probs (list): Lista de arrays com probabilidades
        nomes_modelos (list): Lista com nomes dos modelos
    """
    for nome, y_prob in zip(nomes_modelos, y_probs):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
        plt.title(f'Curva ROC - {nome}', fontsize=15)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/roc/curva_roc_{nome.replace(' ', '_').lower()}.png", dpi=300)
        plt.close()


def visualizar_importancia_features(modelo, feature_names, nome_modelo):
    """
    Visualiza a importância das features para modelos que suportam.

    Args:
        modelo: Modelo treinado
        feature_names (list): Lista com nomes das features
        nome_modelo (str): Nome do modelo
    """
    if hasattr(modelo, 'feature_importances_'):
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices[:15])), importances[indices[:15]], align='center')
        plt.xticks(range(len(indices[:15])), [feature_names[i] for i in indices[:15]], rotation=90)
        plt.title(f'Importância das Features - {nome_modelo}', fontsize=15)
        plt.tight_layout()
        plt.savefig(
            f"{RESULTS_DIR}/graficos/importancia/importancia_features_{nome_modelo.replace(' ', '_').lower()}.png",
            dpi=300)
        plt.close()


# Função para executar várias visualizações de uma vez
def gerar_todos_graficos(df=None, resultados_df=None):
    """
    Gera todos os gráficos disponíveis no sistema.

    Args:
        df (pd.DataFrame, optional): DataFrame com os dados originais
        resultados_df (pd.DataFrame, optional): DataFrame com resultados dos modelos
    """
    if df is not None:
        visualizar_analise_exploratoria(df)
        criar_distribuicao_features_importantes(df)
        LOGGER.info("Gráficos de análise exploratória criados")

    if resultados_df is not None:
        criar_grafico_evolucao_modelos(resultados_df)
        criar_heatmap_metricas_modelos(resultados_df)
        criar_grafico_tempo_treinamento(resultados_df)
        visualizar_comparacao_modelos(resultados_df)
        LOGGER.info("Gráficos de comparação de modelos criados")

    LOGGER.info(f"Todos os gráficos foram salvos em: {RESULTS_DIR}/graficos/")


def exportar_grafico_metrica_barras(valor, nome_metrica_pt, nome_modelo):
    plt.figure(figsize=(6, 6))
    plt.bar([nome_metrica_pt], [valor], color='#2E86C1')
    y_max = 1.0
    if nome_metrica_pt in {'Log Loss', 'Brier Score'} or (isinstance(valor, (int, float)) and valor > 1.0):
        y_max = max(1.0, float(valor) * 1.2)
    plt.ylim(0, y_max)
    plt.title(f'{nome_metrica_pt} - {nome_modelo}')
    plt.ylabel(nome_metrica_pt)
    plt.tight_layout()
    nome_arquivo = {
        'Acurácia': 'acuracia',
        'Precisão': 'precisao',
        'Recall': 'recall',
        'F1-Score': 'f1_score',
        'AUC ROC': 'auc_roc',
        'Especificidade': 'especificidade',
        'Acurácia Balanceada': 'acuracia_balanceada',
        'Log Loss': 'log_loss',
    }.get(nome_metrica_pt, nome_metrica_pt.lower().replace(' ', '_'))
    plt.savefig(f"{RESULTS_DIR}/graficos/metricas/{nome_arquivo}_{nome_modelo}.png", dpi=300)
    plt.close()


def exportar_matriz_confusao(y_true, y_pred, nome_modelo):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([["vn", "fp"], ["fn", "vp"]])
    annot = [[f"{labels[i, j]}\n{cm[i, j]}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues')
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    ax.set_xticklabels(['Negativo', 'Positivo'])
    ax.set_yticklabels(['Negativo', 'Positivo'], rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/confusao/matriz_confusao_{nome_modelo}.png", dpi=300)
    plt.close()


def exportar_metricas_principais(y_true, y_pred, nome_modelo, metricas):
    exportar_grafico_metrica_barras(metricas['accuracy'], 'Acurácia', nome_modelo)
    exportar_grafico_metrica_barras(metricas['precision'], 'Precisão', nome_modelo)
    exportar_grafico_metrica_barras(metricas['recall'], 'Recall', nome_modelo)
    exportar_grafico_metrica_barras(metricas['f1'], 'F1-Score', nome_modelo)
    exportar_matriz_confusao(y_true, y_pred, nome_modelo)


def exportar_metricas_adicionais(y_true, y_pred, nome_modelo, metricas):
    if 'roc_auc' in metricas:
        exportar_grafico_metrica_barras(metricas['roc_auc'], 'AUC ROC', nome_modelo)
    if 'specificity' in metricas:
        exportar_grafico_metrica_barras(metricas['specificity'], 'Especificidade', nome_modelo)
    if 'balanced_accuracy' in metricas:
        exportar_grafico_metrica_barras(metricas['balanced_accuracy'], 'Acurácia Balanceada', nome_modelo)
    if 'log_loss' in metricas:
        exportar_grafico_metrica_barras(metricas['log_loss'], 'Log Loss', nome_modelo)


def visualizar_comparacao_treino_teste(resultados_treino_df, resultados_teste_df, metricas=None):
    """
    Cria gráficos comparativos entre métricas de treino e teste para cada modelo e uma visão agregada.

    Descrição:
        Esta função recebe dois DataFrames (resultados de treino e resultados de teste) contendo as mesmas
        colunas de métricas e a coluna 'modelo'. Ela gera:
         - Para cada modelo, um gráfico em barras comparando as métricas selecionadas (treino vs teste).
         - Um heatmap que mostra a diferença (treino - teste) normalizada para todas as métricas e modelos.
         - Um gráfico de barras agrupadas por métrica com todos os modelos (treino vs teste) para facilitar
           a comparação entre modelos em uma métrica específica.

    Args:
        resultados_treino_df (pd.DataFrame): DataFrame com colunas ['modelo', <metricas...>] contendo métricas de treino.
        resultados_teste_df (pd.DataFrame): DataFrame com colunas ['modelo', <metricas...>] contendo métricas de teste.
        metricas (list, optional): Lista de nomes de colunas de métricas a comparar. Se None, usa uma lista padrão.

    Salidas:
        Arquivos de imagem PNG salvos no diretório de resultados (`RESULTS_DIR/graficos/`) com nomes que indicam
        a comparação treino vs teste por modelo e um heatmap geral.

    Comportamento e validações:
        - Filtra apenas os modelos presentes em ambos os DataFrames.
        - Ignora métricas ausentes em qualquer DataFrame.
        - Garante que os valores estão no intervalo apropriado antes de plotar.
        - Faz logging das ações via LOGGER.
    """

    # Métricas padrão se não fornecidas
    if metricas is None:
        metricas = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Verificar colunas 'modelo'
    if 'modelo' not in resultados_treino_df.columns or 'modelo' not in resultados_teste_df.columns:
        LOGGER.error("DataFrames devem conter a coluna 'modelo'")
        return

    # Encontrar modelos comuns
    modelos_treino = set(resultados_treino_df['modelo'].values)
    modelos_teste = set(resultados_teste_df['modelo'].values)
    modelos_comuns = sorted(list(modelos_treino.intersection(modelos_teste)))

    if not modelos_comuns:
        LOGGER.error('Nenhum modelo comum encontrado entre treino e teste')
        return

    # Filtrar DataFrames para modelos comuns
    tre_df = resultados_treino_df[resultados_treino_df['modelo'].isin(modelos_comuns)].set_index('modelo')
    tes_df = resultados_teste_df[resultados_teste_df['modelo'].isin(modelos_comuns)].set_index('modelo')

    # Determinar métricas disponíveis em ambos
    metricas_disponiveis = [m for m in metricas if m in tre_df.columns and m in tes_df.columns]

    if not metricas_disponiveis:
        LOGGER.error('Nenhuma das métricas solicitadas está disponível em ambos os DataFrames')
        return

    # 1) Para cada modelo: gráfico comparando métricas treino vs teste
    for modelo in modelos_comuns:
        plt.figure(figsize=(12, 6))
        valores_tre = [tre_df.loc[modelo, m] for m in metricas_disponiveis]
        valores_tes = [tes_df.loc[modelo, m] for m in metricas_disponiveis]

        x = np.arange(len(metricas_disponiveis))
        width = 0.35

        plt.bar(x - width/2, valores_tre, width, label='Treino', color='#3498DB')
        plt.bar(x + width/2, valores_tes, width, label='Teste', color='#E74C3C')

        plt.xticks(x, [m.upper() for m in metricas_disponiveis], rotation=45)
        plt.ylim(0, 1)
        plt.title(f'Comparação Treino vs Teste - {modelo}', fontsize=14)
        plt.ylabel('Valor da Métrica')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/comparacao_treino_teste_{modelo.replace(' ', '_').lower()}.png", dpi=300)
        plt.close()

    LOGGER.info('Gráficos individuais Treino vs Teste gerados')

    # 2) Heatmap da diferença (treino - teste)
    diff_df = tre_df[metricas_disponiveis] - tes_df[metricas_disponiveis]

    # Normalizar diferença para visualização (opcional: dividir por treino para escala relativa)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_diff = diff_df.copy()
        denom = tre_df[metricas_disponiveis].abs()
        denom[denom == 0] = 1.0
        norm_diff = diff_df / denom

    plt.figure(figsize=(14, max(6, int(len(modelos_comuns) * 0.5))))
    sns.heatmap(diff_df, annot=True, fmt='.3f', cmap='bwr', center=0, cbar_kws={"shrink": .8})
    plt.title('Diferença das Métricas (Treino - Teste)', fontsize=16)
    plt.xlabel('Métricas')
    plt.ylabel('Modelos')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/heatmap_diferenca_treino_teste.png", dpi=300)
    plt.close()

    LOGGER.info('Heatmap de diferença Treino vs Teste gerado')

    # 3) Gráfico agrupado por métrica mostrando treino x teste para todos os modelos
    for metrica in metricas_disponiveis:
        plt.figure(figsize=(max(10, int(len(modelos_comuns) * 0.6)), 6))
        df_plot = pd.DataFrame({
            'modelo': modelos_comuns,
            'treino': [tre_df.loc[m, metrica] for m in modelos_comuns],
            'teste': [tes_df.loc[m, metrica] for m in modelos_comuns]
        })

        df_melt = df_plot.melt(id_vars='modelo', value_vars=['treino', 'teste'], var_name='conjunto', value_name='valor')
        sns.barplot(x='modelo', y='valor', hue='conjunto', data=df_melt, palette=['#3498DB', '#E74C3C'])
        plt.title(f'Comparação Treino x Teste por Modelo - {metrica.upper()}', fontsize=14)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/comparacao_{metrica}_treino_teste.png", dpi=300)
        plt.close()

    LOGGER.info('Gráficos agrupados por métrica gerados')
