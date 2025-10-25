import os
import time
from config import RESULTS_DIR, LOGGER


def create_cnn_from_hyperparameters(best_hps_dict, input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Reshape, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
    from tensorflow.keras.metrics import AUC, Precision, Recall

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Reshape((input_shape[0], 1)))

    num_conv_layers = best_hps_dict.get('num_conv_layers', 2)

    for i in range(num_conv_layers):
        filters = best_hps_dict.get(f'filters_layer_{i}', 32)
        kernel_size = best_hps_dict.get(f'kernel_size_{i}', 3)
        activation = best_hps_dict.get(f'conv_activation_{i}', 'relu')
        l2_reg = best_hps_dict.get(f'conv_l2_reg_{i}', 0.001)

        model.add(Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            kernel_regularizer=l2(l2_reg),
            padding='same'
        ))

        if best_hps_dict.get(f'conv_batch_norm_{i}', False):
            model.add(BatchNormalization())

        if best_hps_dict.get(f'use_pooling_{i}', True):
            pool_size = best_hps_dict.get(f'pool_size_{i}', 2)
            model.add(MaxPooling1D(pool_size=pool_size))

        dropout_rate = best_hps_dict.get(f'conv_dropout_{i}', 0.3)
        model.add(Dropout(dropout_rate))

    model.add(Flatten())

    num_dense_layers = best_hps_dict.get('num_dense_layers', 2)

    for i in range(num_dense_layers):
        units = best_hps_dict.get(f'dense_units_{i}', 64)
        activation = best_hps_dict.get(f'dense_activation_{i}', 'relu')
        l2_reg = best_hps_dict.get(f'dense_l2_reg_{i}', 0.001)

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        if best_hps_dict.get(f'dense_batch_norm_{i}', False):
            model.add(BatchNormalization())

        dropout_rate = best_hps_dict.get(f'dense_dropout_{i}', 0.3)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = best_hps_dict.get('optimizer', 'adam')
    learning_rate = best_hps_dict.get('learning_rate', 0.001)

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


def retrain_final_cnn(best_hps, x_train_full, y_train_full, model_name="CNN_Final"):
    from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("RETREINANDO MELHOR CNN COM TODOS OS DADOS")
    LOGGER.info("=" * 80)
    LOGGER.info(f"  - Train shape: {x_train_full.shape}")

    input_shape = (x_train_full.shape[1],)

    if hasattr(best_hps, 'values'):
        from cnn_tuning import create_cnn_from_best_hps
        final_model = create_cnn_from_best_hps(best_hps, input_shape)
    else:
        final_model = create_cnn_from_hyperparameters(best_hps, input_shape)

    epoch_logger = LambdaCallback(
        on_epoch_end=lambda epoch, logs: LOGGER.info(
            f"Época {epoch+1}/150 - loss: {logs['loss']:.4f} - "
            f"accuracy: {logs.get('accuracy', 0):.4f} - "
            f"precision: {logs.get('precision', 0):.4f}"
        ) if (epoch + 1) % 5 == 0 else None
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    batch_size = best_hps.get('batch_size') if hasattr(best_hps, 'get') else 64
    LOGGER.info(f"Usando batch_size: {batch_size}")
    LOGGER.info("Iniciando treinamento final da CNN...")

    retrain_start = time.time()
    history = final_model.fit(
        x_train_full, y_train_full,
        epochs=150,
        batch_size=batch_size,
        callbacks=[early_stop, epoch_logger],
        verbose=0
    )

    retrain_time = time.time() - retrain_start
    LOGGER.info(f"Treinamento final da CNN concluído em {retrain_time/60:.2f} minutos")
    LOGGER.info(f"Épocas executadas: {len(history.history['loss'])}")

    final_model_path = os.path.join(RESULTS_DIR, "modelos", f"{model_name}.keras")
    final_model.save(final_model_path)
    LOGGER.info(f"Modelo CNN final salvo em: {final_model_path}")

    return final_model


def load_best_cnn_model(load_model):
    tuned_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Tuned_Final.keras")
    bayesian_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Bayesian_Final.keras")

    if os.path.exists(bayesian_path):
        LOGGER.info("Carregando modelo CNN Bayesian...")
        return load_model(bayesian_path), "CNN_Bayesian"
    elif os.path.exists(tuned_path):
        LOGGER.info("Carregando modelo CNN Tuned...")
        return load_model(tuned_path), "CNN_Tuned"

    return None, None

