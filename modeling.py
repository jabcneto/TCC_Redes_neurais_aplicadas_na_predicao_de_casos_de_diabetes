# modeling.py
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Activation, PReLU, LeakyReLU,
    Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from config import LOGGER


def criar_modelo_mlp(input_shape, learning_rate=0.001, regularization=0.001):
    """Cria um modelo MLP (Perceptron Multicamadas) para classificação binária."""
    LOGGER.info("Criando modelo MLP.")
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    model.summary()
    return model


def criar_modelo_cnn(input_shape, learning_rate=0.001, regularization=0.001):
    """Cria um modelo CNN 1D para classificação binária de dados tabulares."""
    LOGGER.info("Criando modelo CNN.")

    # Garante que o input_shape seja 3D para a CNN
    if len(input_shape) == 1:
        actual_input_shape = (input_shape[0], 1)
    else:
        actual_input_shape = input_shape

    model = Sequential([
        Input(shape=actual_input_shape),
        Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(regularization), activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    model.summary()
    return model


def criar_modelo_hibrido(input_shape, learning_rate=0.001, regularization=0.001):
    """Cria um modelo híbrido CNN-LSTM para classificação binária."""
    LOGGER.info("Criando modelo híbrido CNN-LSTM.")

    # Garante que o input_shape seja 3D
    if len(input_shape) == 1:
        actual_input_shape = (input_shape[0], 1)
    else:
        actual_input_shape = input_shape

    inputs = Input(shape=actual_input_shape)

    # Camadas CNN
    x = Conv1D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(regularization), activation='relu')(
        inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Camada LSTM
    x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)

    # Camadas Densas
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    model.summary()
    return model


def get_classic_models(random_state):
    """Retorna um dicionário de modelos clássicos de ML."""
    models = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    }
    return models