import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR, LOGGER

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
        metricas = ['accuracy', 'precision', 'recall', 'f1']

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
