import os
import json
import time
import numpy as np
from config import RESULTS_DIR, LOGGER


def build_tunable_cnn(hp):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Reshape, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
    from tensorflow.keras.metrics import AUC, Precision, Recall

    input_dim = hp.get('input_dim')

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Reshape((input_dim, 1)))

    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3, default=2)

    for i in range(num_conv_layers):
        filters = hp.Int(f'filters_layer_{i}', min_value=16, max_value=128, step=16, default=32)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5, 7], default=3)
        activation = hp.Choice(f'conv_activation_{i}', values=['relu', 'elu', 'selu'], default='relu')
        l2_reg = hp.Float(f'conv_l2_reg_{i}', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)

        model.add(Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            padding='same'
        ))

        if hp.Boolean(f'conv_batch_norm_{i}', default=False):
            model.add(BatchNormalization())

        if hp.Boolean(f'use_pooling_{i}', default=True):
            pool_size = hp.Choice(f'pool_size_{i}', values=[2, 3], default=2)
            model.add(MaxPooling1D(pool_size=pool_size))

        dropout_rate = hp.Float(f'conv_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
        model.add(Dropout(dropout_rate))

    model.add(Flatten())

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, default=2)

    for i in range(num_dense_layers):
        units = hp.Int(f'dense_units_{i}', min_value=32, max_value=256, step=32, default=64)
        activation = hp.Choice(f'dense_activation_{i}', values=['relu', 'elu', 'selu'], default='relu')
        l2_reg = hp.Float(f'dense_l2_reg_{i}', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        if hp.Boolean(f'dense_batch_norm_{i}', default=False):
            model.add(BatchNormalization())

        dropout_rate = hp.Float(f'dense_dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop', 'sgd'], default='adam')
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )

    return model


def tune_cnn_hyperparameters(x_train, y_train, x_val, y_val, max_trials=50, executions_per_trial=2):
    import keras_tuner as kt
    from tensorflow.keras.callbacks import EarlyStopping

    tuning_dir = os.path.join(RESULTS_DIR, 'tuning', 'cnn_results')
    os.makedirs(tuning_dir, exist_ok=True)

    def model_builder(hp):
        hp.Fixed('input_dim', value=x_train.shape[1])
        return build_tunable_cnn(hp)

    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective('val_precision', direction='max'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=tuning_dir,
        project_name='cnn_tuning',
        overwrite=False
    )

    early_stop = EarlyStopping(monitor='val_precision', patience=15, mode='max', restore_best_weights=True)

    LOGGER.info("Iniciando busca de hiperparâmetros para CNN...")
    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    _save_best_trial_config(tuner, best_hps)

    return best_model, best_hps, tuner


def bayesian_tune_cnn(x_train, y_train, x_val, y_val, max_trials=30, executions_per_trial=2):
    import keras_tuner as kt
    from tensorflow.keras.callbacks import EarlyStopping

    tuning_dir = os.path.join(RESULTS_DIR, 'tuning', 'cnn_bayesian_results')
    os.makedirs(tuning_dir, exist_ok=True)

    def model_builder(hp):
        hp.Fixed('input_dim', value=x_train.shape[1])
        return build_tunable_cnn(hp)

    tuner = kt.BayesianOptimization(
        hypermodel=model_builder,
        objective=kt.Objective('val_precision', direction='max'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=tuning_dir,
        project_name='cnn_bayesian_tuning',
        overwrite=False
    )

    early_stop = EarlyStopping(monitor='val_precision', patience=15, mode='max', restore_best_weights=True)

    LOGGER.info("Iniciando busca bayesiana de hiperparâmetros para CNN...")
    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    _save_bayesian_config(tuner, best_hps)

    return best_model, best_hps, tuner


def create_cnn_from_best_hps(best_hps, input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Reshape, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
    from tensorflow.keras.metrics import AUC, Precision, Recall

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Reshape((input_shape[0], 1)))

    num_conv_layers = best_hps.get('num_conv_layers')

    for i in range(num_conv_layers):
        filters = best_hps.get(f'filters_layer_{i}')
        kernel_size = best_hps.get(f'kernel_size_{i}')
        activation = best_hps.get(f'conv_activation_{i}')
        l2_reg = best_hps.get(f'conv_l2_reg_{i}')

        model.add(Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            padding='same'
        ))

        if best_hps.get(f'conv_batch_norm_{i}'):
            model.add(BatchNormalization())

        if best_hps.get(f'use_pooling_{i}'):
            pool_size = best_hps.get(f'pool_size_{i}')
            model.add(MaxPooling1D(pool_size=pool_size))

        dropout_rate = best_hps.get(f'conv_dropout_{i}')
        model.add(Dropout(dropout_rate))

    model.add(Flatten())

    num_dense_layers = best_hps.get('num_dense_layers')

    for i in range(num_dense_layers):
        units = best_hps.get(f'dense_units_{i}')
        activation = best_hps.get(f'dense_activation_{i}')
        l2_reg = best_hps.get(f'dense_l2_reg_{i}')

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        if best_hps.get(f'dense_batch_norm_{i}'):
            model.add(BatchNormalization())

        dropout_rate = best_hps.get(f'dense_dropout_{i}')
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = best_hps.get('optimizer')
    learning_rate = best_hps.get('learning_rate')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )

    return model


def analyze_cnn_tuning_results(tuner, top_n=10):
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"TOP {top_n} MELHORES CONFIGURAÇÕES DE CNN")
    LOGGER.info("=" * 80)

    best_trials = tuner.oracle.get_best_trials(num_trials=top_n)

    for idx, trial in enumerate(best_trials, 1):
        LOGGER.info(f"\n--- Trial #{trial.trial_id} (Rank {idx}) ---")
        LOGGER.info(f"Val Precision: {trial.score:.4f}")

        hps = trial.hyperparameters.values
        LOGGER.info(f"  Camadas Conv: {hps.get('num_conv_layers')}")
        LOGGER.info(f"  Camadas Dense: {hps.get('num_dense_layers')}")
        LOGGER.info(f"  Optimizer: {hps.get('optimizer')}")
        LOGGER.info(f"  Learning Rate: {hps.get('learning_rate'):.6f}")


def _save_best_trial_config(tuner, best_hps):
    tuning_results_dir = os.path.join(RESULTS_DIR, 'tuning', 'cnn_results')
    os.makedirs(tuning_results_dir, exist_ok=True)

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    config = {
        'trial_number': best_trial.trial_id,
        'best_val_precision': float(best_trial.score),
        'hyperparameters': best_hps.values
    }

    config_path = os.path.join(tuning_results_dir, 'best_trial_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    LOGGER.info(f"\nConfiguração do melhor trial salva em: {config_path}")


def _save_bayesian_config(tuner, best_hps):
    tuning_results_dir = os.path.join(RESULTS_DIR, 'tuning', 'cnn_bayesian_results')
    os.makedirs(tuning_results_dir, exist_ok=True)

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    config = {
        'trial_number': best_trial.trial_id,
        'best_val_precision': float(best_trial.score),
        'hyperparameters': best_hps.values
    }

    config_path = os.path.join(tuning_results_dir, 'bayesian_best_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    LOGGER.info(f"\nConfiguração bayesiana salva em: {config_path}")

