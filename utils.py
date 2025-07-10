# config.py
import os
import numpy as np
import tensorflow as tf
from config import configurar_logging, criar_diretorios_projeto

# --- CONSTANTES GLOBAIS ---
DATASET_PATH = "diabetes_prediction_dataset.csv"
RANDOM_STATE = 42

# --- CONFIGURAÇÕES DE REPRODUTIBILIDADE ---
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- MAPEAMENTO DE COLUNAS ---
MAPEAMENTO_COLUNAS_PT = {
    'gender': 'gênero',
    'age': 'idade',
    'hypertension': 'hipertensão',
    'heart_disease': 'doença cardíaca',
    'smoking_history': 'histórico de tabagismo',
    'bmi': 'IMC',
    'HbA1c_level': 'nível de HbA1c',
    'blood_glucose_level': 'nível de glicose no sangue',
    'diabetes': 'diabetes'
}

# --- LOGGER ---
LOGGER = configurar_logging()