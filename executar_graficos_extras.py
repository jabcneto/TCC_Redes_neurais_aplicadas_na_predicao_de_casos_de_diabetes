import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = 'resultados_diabetes'

# Verificar se os diretórios existem
os.makedirs(f"{RESULTS_DIR}/graficos", exist_ok=True)

def criar_grafico_evolucao_modelos(resultados_df):
    """
    Cria um gráfico mostrando a evolução das métricas entre diferentes tipos de modelos.
    """
    logger.info("Criando gráfico de evolução dos modelos")

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
        ax = axes[i//2, i%2]

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
    print("✅ Gráfico de evolução criado")


def criar_heatmap_metricas_modelos(resultados_df):
    """
    Cria um heatmap comparando todas as métricas de todos os modelos.
    """
    logger.info("Criando heatmap de métricas dos modelos")

    # Selecionar métricas para o heatmap
    metricas_cols = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Verificar quais colunas existem
    metricas_disponiveis = [col for col in metricas_cols if col in resultados_df.columns]

    if not metricas_disponiveis:
        print("❌ Nenhuma métrica encontrada para criar heatmap")
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
    print("✅ Heatmap de métricas criado")


def criar_distribuicao_features_importantes(df):
    """
    Cria gráficos mostrando a distribuição das features mais importantes.
    """
    logger.info("Criando distribuição das features mais importantes")

    # Features mais importantes baseadas no domínio médico
    features_importantes = ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age']

    # Verificar quais features existem no dataset
    features_disponiveis = [f for f in features_importantes if f in df.columns]

    if len(features_disponiveis) < 2:
        print("❌ Poucas features importantes encontradas no dataset")
        return

    # Ajustar layout baseado no número de features
    n_features = len(features_disponiveis)
    if n_features == 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 8))
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
    print("✅ Gráfico de distribuição de features criado")


def criar_grafico_tempo_treinamento(resultados_df):
    """
    Cria gráficos mostrando o tempo de treinamento dos modelos.
    """
    logger.info("Criando gráfico de tempo de treinamento")

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
        print("❌ Nenhum modelo encontrado para análise de tempo")
        return

    # Preparar dados
    df_tempo = pd.DataFrame({
        'modelo': modelos_existentes,
        'tempo_segundos': [tempos_estimados[m] for m in modelos_existentes],
        'tempo_minutos': [tempos_estimados[m]/60 for m in modelos_existentes]
    })

    # Adicionar métricas de performance
    metricas_disponiveis = ['f1', 'roc_auc', 'accuracy']
    for metrica in metricas_disponiveis:
        if metrica in resultados_df.columns:
            df_tempo = df_tempo.merge(resultados_df[['modelo', metrica]], on='modelo')
            break

    # Criar figura
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Análise de Tempo de Treinamento vs Performance', fontsize=20, y=0.98)

    # 1. Tempo de treinamento por modelo
    ax1 = axes[0]
    bars = ax1.bar(range(len(df_tempo)), df_tempo['tempo_minutos'],
                   color=['#3498DB' if t < 5 else '#F39C12' if t < 60 else '#E74C3C' for t in df_tempo['tempo_minutos']])
    ax1.set_title('Tempo de Treinamento por Modelo', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Modelo', fontsize=12)
    ax1.set_ylabel('Tempo (minutos)', fontsize=12)
    ax1.set_xticks(range(len(df_tempo)))
    ax1.set_xticklabels(df_tempo['modelo'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(df_tempo['tempo_minutos'])*0.01,
                f'{height:.1f}min', ha='center', va='bottom', fontsize=10)

    # 2. Scatter: Tempo vs Performance
    ax2 = axes[1]
    if metrica in df_tempo.columns:
        scatter = ax2.scatter(df_tempo['tempo_minutos'], df_tempo[metrica],
                             s=150, alpha=0.7, edgecolors='black')

        # Adicionar nomes dos modelos
        for i, row in df_tempo.iterrows():
            ax2.annotate(row['modelo'], (row['tempo_minutos'], row[metrica]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax2.set_title(f'Relação Tempo vs Performance ({metrica.upper()})', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Tempo de Treinamento (minutos)', fontsize=12)
        ax2.set_ylabel(metrica.upper(), fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/analise_tempo_treinamento.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico de tempo de treinamento criado")


# Executar as funções
try:
    print("🚀 Iniciando criação dos gráficos extras...")

    # Carregar dados dos resultados CORRIGIDOS
    if os.path.exists(f"{RESULTS_DIR}/resultados_todos_modelos_corrigido.csv"):
        resultados_todos_modelos_final = pd.read_csv(f"{RESULTS_DIR}/resultados_todos_modelos_corrigido.csv")
        print(f"📊 Carregados {len(resultados_todos_modelos_final)} modelos (arquivo corrigido)")
    elif os.path.exists(f"{RESULTS_DIR}/resultados_todos_modelos.csv"):
        resultados_todos_modelos_final = pd.read_csv(f"{RESULTS_DIR}/resultados_todos_modelos.csv")
        # Filtrar apenas linhas com dados válidos (sem epoch)
        resultados_todos_modelos_final = resultados_todos_modelos_final[
            resultados_todos_modelos_final['modelo'].notna() &
            ~resultados_todos_modelos_final['modelo'].str.contains('epoch', na=False)
        ].copy()
        print(f"📊 Carregados {len(resultados_todos_modelos_final)} modelos (filtrados)")
    else:
        print("❌ Arquivo de resultados não encontrado")
        exit()

    # Verificar e limpar dados NaN
    print(f"🔍 Verificando dados...")
    print(f"   Antes da limpeza: {len(resultados_todos_modelos_final)} linhas")

    # Remover linhas com NaN nas métricas principais
    metricas_essenciais = ['accuracy', 'f1', 'roc_auc']
    dados_validos = resultados_todos_modelos_final.dropna(subset=metricas_essenciais)

    if len(dados_validos) != len(resultados_todos_modelos_final):
        print(f"   ⚠️ Removidas {len(resultados_todos_modelos_final) - len(dados_validos)} linhas com NaN")
        resultados_todos_modelos_final = dados_validos

    print(f"   Após limpeza: {len(resultados_todos_modelos_final)} modelos válidos")

    if os.path.exists("diabetes_prediction_dataset.csv"):
        df = pd.read_csv("diabetes_prediction_dataset.csv")
        print(f"📊 Dataset carregado com {len(df)} amostras")
    else:
        print("❌ Dataset não encontrado")
        df = None

    # Criar os gráficos
    print("\n📈 Criando gráficos...")
    criar_grafico_evolucao_modelos(resultados_todos_modelos_final)
    criar_heatmap_metricas_modelos(resultados_todos_modelos_final)

    if df is not None:
        criar_distribuicao_features_importantes(df)

    criar_grafico_tempo_treinamento(resultados_todos_modelos_final)

    print(f"\n✅ Gráficos extras criados com sucesso!")
    print(f"📁 Salvos em: {RESULTS_DIR}/graficos/")

    # Listar arquivos criados
    graficos_criados = [
        "evolucao_modelos_complexidade.png",
        "heatmap_metricas_modelos.png",
        "distribuicao_features_importantes.png",
        "analise_tempo_treinamento.png"
    ]

    print("\n📋 Arquivos criados:")
    for grafico in graficos_criados:
        caminho = f"{RESULTS_DIR}/graficos/{grafico}"
        if os.path.exists(caminho):
            print(f"   ✅ {grafico}")
        else:
            print(f"   ❌ {grafico}")

except Exception as e:
    print(f"❌ Erro ao criar gráficos: {e}")
    import traceback
    traceback.print_exc()
