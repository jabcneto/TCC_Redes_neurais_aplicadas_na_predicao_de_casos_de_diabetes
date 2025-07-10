# training.py
import pickle
import time
import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from utils import RESULTS_DIR
from config import LOGGER


def criar_callbacks_pt(nome_modelo, paciencia=15, monitor='val_auc'):
    """Cria callbacks para o treinamento de modelos Keras."""
    log_dir = os.path.join(RESULTS_DIR, "logs", f"{nome_modelo}_{time.strftime('%Y%m%d-%H%M%S')}")
    best_model_path = os.path.join(RESULTS_DIR, "modelos", f"{nome_modelo}_best.keras")

    # Determina o modo (max ou min) com base no nome da métrica monitorada
    if 'acc' in monitor or 'auc' in monitor:
        mode = 'max'
    else:
        mode = 'min'

    LOGGER.info(f"Callback monitorando '{monitor}' no modo '{mode}'.")

    return [
        EarlyStopping(
            monitor=monitor,
            patience=paciencia,
            verbose=1,
            mode=mode,  # <-- CORREÇÃO APLICADA AQUI
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            save_best_only=True,
            monitor=monitor,
            mode=mode,  # <-- CORREÇÃO APLICADA AQUI
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=paciencia // 2,
            min_lr=1e-6,
            mode=mode,  # <-- CORREÇÃO APLICADA AQUI
            verbose=1
        ),
        CSVLogger(filename=os.path.join(RESULTS_DIR, "history", f"{nome_modelo}_history.csv")),
        TensorBoard(log_dir=log_dir)
    ]


def treinar_modelo_keras_pt(model, x_train, y_train, x_val, y_val, nome_modelo, epochs=100, batch_size=128):
    """
    Treina um modelo Keras.
    A barra de progresso já é fornecida pelo Keras (verbose=1).
    """
    LOGGER.info(f"Iniciando treinamento do modelo: {nome_modelo}")
    callbacks = criar_callbacks_pt(nome_modelo, monitor='val_auc')

    if "cnn" in nome_modelo.lower() or "hibrido" in nome_modelo.lower():
        if len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1)
            x_val = np.expand_dims(x_val, axis=-1)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    model.save(os.path.join(RESULTS_DIR, "modelos", f"{nome_modelo}_final.keras"))
    return model, history


def treinar_modelos_classicos_pt(models, x_train, y_train):
    """Treina uma lista de modelos clássicos com uma barra de progresso."""
    trained_models = {}

    LOGGER.info("Iniciando treinamento dos modelos clássicos...")
    for name, model in tqdm(models.items(), desc="Treinando modelos clássicos"):
        if hasattr(model, 'verbose'):
            model.verbose = 0
        if hasattr(model, 'verbosity'):
            model.verbosity = 0

        model.fit(x_train, y_train)

        with open(os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl"), 'wb') as f:
            pickle.dump(model, f)

        trained_models[name] = model

    LOGGER.info("Treinamento dos modelos clássicos concluído.")
    return trained_models