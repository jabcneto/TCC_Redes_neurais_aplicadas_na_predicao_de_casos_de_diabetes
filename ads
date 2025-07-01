def analisar_dados(df):
    """
    Realiza análise exploratória detalhada dos dados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        
    Returns:
        dict: Dicionário com estatísticas e informações da análise
    """
    logger.info("Realizando análise exploratória dos dados")
    
    # Estatísticas básicas
    estatisticas = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'target_distribution': df['diabetes'].value_counts(normalize=True).to_dict(),
        'numeric_stats': df.describe().to_dict(),
    }
    
    # Análise de variáveis categóricas
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    estatisticas['categorical_counts'] = {col: df[col].value_counts().to_dict() for col in cat_cols}
    
    # Detecção de outliers (usando IQR)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    outliers = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        }
    estatisticas['outliers'] = outliers
    
    # Correlações
    # Garante que apenas colunas numéricas sejam usadas para correlação
    numeric_df = df.select_dtypes(include=np.number)
    if 'diabetes' in numeric_df.columns:
        estatisticas['correlations'] = numeric_df.corr()['diabetes'].to_dict()
    else:
        estatisticas['correlations'] = {}
        logger.warning("Coluna 'diabetes' não encontrada ou não é numérica para cálculo de correlação.")

    
    # Visualizações
    visualizar_analise_exploratoria(df)
    
    return estatisticas

def visualizar_analise_exploratoria(df):
    """
    Cria visualizações para análise exploratória dos dados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
    """
    logger.info("Gerando visualizações para análise exploratória")
    
    # Configuração para visualizações
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    
    # 1. Distribuição da variável alvo
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='diabetes', data=df, palette=['#3498db', '#e74c3c'])
    plt.title('Distribuição da Variável Alvo (Diabetes)', fontsize=15)
    plt.xlabel('Diabetes', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    
    # Adicionar percentagens
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 5,
                f'{height} ({height/total:.1%})',
                ha="center", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao_target.png", dpi=300)
    plt.close()
    
    # 2. Distribuição das variáveis numéricas por status de diabetes
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'diabetes' in num_cols:
        num_cols.remove('diabetes') # Remove a variável alvo se estiver presente
    
    if num_cols:
        plt.figure(figsize=(15, 5 * ((len(num_cols) + 1) // 2)))
        for i, col in enumerate(num_cols):
            plt.subplot((len(num_cols) + 1) // 2, 2, i+1)
            sns.histplot(data=df, x=col, hue='diabetes', kde=True, bins=30, 
                         palette=['#3498db', '#e74c3c'], alpha=0.6)
            plt.title(f'Distribuição de {col} por Status de Diabetes', fontsize=13)
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Contagem', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao_variaveis_numericas.png", dpi=300)
        plt.close()

        # 3. Boxplots para variáveis numéricas
        plt.figure(figsize=(15, 5 * ((len(num_cols) + 1) // 2)))
        for i, col in enumerate(num_cols):
            plt.subplot((len(num_cols) + 1) // 2, 2, i+1)
            sns.boxplot(x='diabetes', y=col, data=df, palette=['#3498db', '#e74c3c'])
            plt.title(f'{col} por Status de Diabetes', fontsize=13)
            plt.xlabel('Diabetes', fontsize=11)
            plt.ylabel(col, fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/boxplots_variaveis_numericas.png", dpi=300)
        plt.close()
    
    # 4. Matriz de correlação
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        corr_matrix = numeric_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    square=True, linewidths=0.5)
        plt.title('Matriz de Correlação', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/matriz_correlacao.png", dpi=300)
        plt.close()
    
    # 5. Pairplot para variáveis numéricas (se houver)
    if num_cols and 'diabetes' in df.columns:
        sns.pairplot(df[num_cols + ['diabetes']], hue='diabetes', 
                     palette=['#3498db', '#e74c3c'], diag_kind='kde')
        plt.suptitle('Pairplot de Variáveis Numéricas', y=1.02, fontsize=16)
        plt.savefig(f"{RESULTS_DIR}/graficos/pairplot_variaveis_numericas.png", dpi=300)
        plt.close()
    
    # 6. Contagem de variáveis categóricas
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if cat_cols and 'diabetes' in df.columns:
        plt.figure(figsize=(15, 5 * len(cat_cols)))
        
        for i, col in enumerate(cat_cols):
            plt.subplot(len(cat_cols), 1, i+1)
            sns.countplot(x=col, hue='diabetes', data=df, palette=['#3498db', '#e74c3c'])
            plt.title(f'Distribuição de {col} por Status de Diabetes', fontsize=13)
            plt.xlabel(col, fontsize=11)
            plt.ylabel('Contagem', fontsize=11)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/distribuicao_variaveis_categoricas.png", dpi=300)
        plt.close()
    
    # 7. Relação entre HbA1c e glicose com diabetes
    if 'HbA1c_level' in df.columns and 'blood_glucose_level' in df.columns and 'diabetes' in df.columns:
        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(data=df, x='HbA1c_level', y='blood_glucose_level', 
                                 hue='diabetes', palette=['#3498db', '#e74c3c'], 
                                 s=80, alpha=0.7)
        plt.axvline(x=6.5, color='red', linestyle='--', label='Limiar HbA1c (6.5%)')
        plt.axhline(y=126, color='green', linestyle='--', label='Limiar Glicose (126 mg/dL)')
        
        # Adicionar anotações para os quadrantes
        plt.text(7.5, 200, 'Alto riscon(HbA1c alto, Glicose alta)', fontsize=12, ha='center')
        plt.text(5.5, 200, 'Risco moderadon(HbA1c normal, Glicose alta)', fontsize=12, ha='center')
        plt.text(7.5, 100, 'Risco moderadon(HbA1c alto, Glicose normal)', fontsize=12, ha='center')
        plt.text(5.5, 100, 'Baixo riscon(HbA1c normal, Glicose normal)', fontsize=12, ha='center')
        
        plt.title('Relação entre HbA1c e Glicose no Sangue', fontsize=15)
        plt.xlabel('Nível de HbA1c (%)', fontsize=12)
        plt.ylabel('Nível de Glicose (mg/dL)', fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/graficos/relacao_hba1c_glicose.png", dpi=300)
        plt.close()