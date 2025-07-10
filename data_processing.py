# data_processing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import LOGGER, MAPEAMENTO_COLUNAS_PT, RANDOM_STATE
from evaluation import visualizar_analise_exploratoria_dados


def renomear_colunas_para_portugues(df):
    return df.rename(columns=MAPEAMENTO_COLUNAS_PT)


def carregar_dados(caminho_arquivo):
    LOGGER.info(f"Carregando dados de {caminho_arquivo}")
    try:
        dataframe = pd.read_csv(caminho_arquivo)
        dataframe = renomear_colunas_para_portugues(dataframe)
        LOGGER.info(f"Dataset carregado. Formato: {dataframe.shape}")
        return dataframe
    except Exception as e:
        LOGGER.error(f"Erro ao carregar o dataset: {e}")
        return None


def analisar_dados(df):
    LOGGER.info("Realizando análise exploratória dos dados.")
    if 'gênero' in df.columns and 'Other' in df['gênero'].unique():
        df = df[df['gênero'] != 'Other'].copy()
    visualizar_analise_exploratoria_dados(df)
    return df


def tratar_outliers_iqr(df, colunas_numericas, fator=1.5):
    df_tratado = df.copy()
    for col in colunas_numericas:
        Q1 = df_tratado[col].quantile(0.25)
        Q3 = df_tratado[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - fator * IQR
        upper_bound = Q3 + fator * IQR
        df_tratado[col] = df_tratado[col].clip(lower=lower_bound, upper=upper_bound)
    return df_tratado


def pre_processar_dados(dataframe, test_size=0.2, val_size=0.1):
    LOGGER.info("Iniciando pré-processamento dos dados.")
    x = dataframe.drop(columns=['diabetes'])
    y = dataframe['diabetes']

    colunas_numericas = x.select_dtypes(include=['int64', 'float64']).columns
    colunas_categoricas = x.select_dtypes(include=['object']).columns

    x = tratar_outliers_iqr(x, colunas_numericas)

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(x[colunas_categoricas])
    encoded_cols = encoder.get_feature_names_out(colunas_categoricas)
    x_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=x.index)
    x_final = pd.concat([x[colunas_numericas], x_encoded], axis=1)
    feature_names = x_final.columns.tolist()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_final)

    smote = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = smote.fit_resample(x_scaled, y)

    x_train, x_temp, y_train, y_temp = train_test_split(x_resampled, y_resampled, test_size=test_size + val_size,
                                                        random_state=RANDOM_STATE, stratify=y_resampled)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                    random_state=RANDOM_STATE, stratify=y_temp)

    LOGGER.info(f"Divisão dos dados: Treino={x_train.shape}, Validação={x_val.shape}, Teste={x_test.shape}")

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler, encoder, feature_names