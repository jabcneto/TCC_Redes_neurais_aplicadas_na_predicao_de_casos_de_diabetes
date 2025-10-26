import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from config import LOGGER


def criar_modelo_mlp_pt(input_shape, learning_rate=0.0005, regularization=0.01):
    """Cria um modelo MLP (Perceptron Multicamadas) para classificação binária."""
    LOGGER.info("Criando modelo MLP.")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import Precision, Recall, AUC, PrecisionAtRecall

    model = Sequential([
        Input(shape=input_shape),
        Dense(64, kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(16, kernel_regularizer=l2(regularization), activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), AUC(curve='PR', name='pr_auc'), Precision(name='precision'), Recall(name='recall'), PrecisionAtRecall(0.80, name='precision_at_recall_80')]
    )
    model.summary()
    return model


def criar_modelo_cnn_pt(input_shape, learning_rate=0.0005, regularization=0.01):
    """Cria um modelo CNN 1D para classificação binária de dados tabulares."""
    LOGGER.info("Criando modelo CNN.")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input,
        Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import Precision, Recall, AUC, PrecisionAtRecall

    if len(input_shape) == 1:
        actual_input_shape = (input_shape[0], 1)
    else:
        actual_input_shape = input_shape
    model = Sequential([
        Input(shape=actual_input_shape),
        Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        Dense(16, kernel_regularizer=l2(regularization), activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), AUC(curve='PR', name='pr_auc'), Precision(name='precision'), Recall(name='recall'), PrecisionAtRecall(0.80, name='precision_at_recall_80')]
    )
    model.summary()
    return model


def criar_modelo_hibrido_pt(input_shape, learning_rate=0.0005, regularization=0.01):
    """Cria um modelo híbrido CNN-LSTM para classificação binária."""
    LOGGER.info("Criando modelo híbrido CNN-LSTM.")
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input,
        Conv1D, Bidirectional, LSTM, GlobalAveragePooling1D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import Precision, Recall, AUC, PrecisionAtRecall

    if len(input_shape) == 1:
        actual_input_shape = (input_shape[0], 1)
    else:
        actual_input_shape = input_shape
    inputs = Input(shape=actual_input_shape)
    x = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(regularization), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, kernel_regularizer=l2(regularization), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, kernel_regularizer=l2(regularization), activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), AUC(curve='PR', name='pr_auc'), Precision(name='precision'), Recall(name='recall'), PrecisionAtRecall(0.80, name='precision_at_recall_80')]
    )
    model.summary()
    return model


def obter_modelos_classicos(random_state):
    models = {
        'Regressão Logística': LogisticRegression(
            random_state=random_state,
            C=0.1,
            max_iter=1000,
            penalty='l2'
        ),
        'Random Forest': RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        'XGBoost': XGBClassifier(
            random_state=random_state,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
    }
    return models