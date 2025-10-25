import os
import json
import pandas as pd
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.metrics import Precision, Recall
from config import RESULTS_DIR, LOGGER


class TuningProgressCallback(keras.callbacks.Callback):
    def __init__(self, trial_number, total_trials):
        super().__init__()
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.best_val_precision = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_precision = logs.get('val_precision', 0)
        val_loss = logs.get('val_loss', 0)

        if val_precision > self.best_val_precision:
            self.best_val_precision = val_precision
            LOGGER.info(f"  Trial {self.trial_number}/{self.total_trials} - Época {epoch+1}: "
                       f"val_precision melhorou para {val_precision:.4f}")


class CustomTuner(kt.BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_count = 0
        self.best_precision_so_far = 0
        self.total_trials = kwargs.get('max_trials', 50)
        self.trials_data = []

    def run_trial(self, trial, *args, **kwargs):
        self.trial_count += 1

        LOGGER.info("=" * 80)
        LOGGER.info(f"INICIANDO TRIAL {self.trial_count}/{self.total_trials}")
        LOGGER.info("=" * 80)

        hp = trial.hyperparameters
        LOGGER.info(f"Configuração do Trial {self.trial_count}:")

        num_layers = hp.values.get('num_layers', 0)
        LOGGER.info(f"  - Camadas: {num_layers}")
        LOGGER.info(f"  - Otimizador: {hp.values.get('optimizer', 'N/A')}")
        learning_rate = hp.values.get('learning_rate', 0)
        if learning_rate:
            LOGGER.info(f"  - Learning Rate: {learning_rate:.6f}")

        for i in range(num_layers):
            units = hp.values.get(f'units_layer_{i}', 'N/A')
            activation = hp.values.get(f'activation_{i}', 'N/A')
            dropout = hp.values.get(f'dropout_{i}', 0)
            LOGGER.info(f"  - Camada {i+1}: {units} neurônios, "
                       f"ativação={activation}, "
                       f"dropout={dropout:.2f}")

        try:
            result = super().run_trial(trial, *args, **kwargs)
        except Exception as e:
            LOGGER.error(f"Erro durante trial {self.trial_count}: {e}")
            import traceback
            LOGGER.error(f"Traceback: {traceback.format_exc()}")
            raise

        trial_data = {
            'trial_id': trial.trial_id,
            'trial_number': self.trial_count,
            'status': trial.status,
            'hyperparameters': dict(hp.values)
        }

        if hasattr(trial, 'metrics') and trial.metrics:
            try:
                precision_history = trial.metrics.get_history('val_precision')
                if precision_history:
                    best_precision = max(precision_history)
                    trial_data['best_val_precision'] = best_precision
                    trial_data['final_val_precision'] = precision_history[-1] if precision_history else 0
                    trial_data['precision_history'] = precision_history

                    LOGGER.info(f"\nTrial {self.trial_count} concluído!")
                    LOGGER.info(f"  Melhor val_precision deste trial: {best_precision:.4f}")

                    if best_precision > self.best_precision_so_far:
                        self.best_precision_so_far = best_precision
                        LOGGER.info(f"  ★ NOVO RECORDE! Melhor precisão até agora: {self.best_precision_so_far:.4f}")
                    else:
                        LOGGER.info(f"  Melhor precisão global: {self.best_precision_so_far:.4f}")
                else:
                    LOGGER.warning(f"Trial {self.trial_count}: val_precision history está vazio")
                    trial_data['best_val_precision'] = 0
            except (ValueError, KeyError) as e:
                LOGGER.warning(f"Não foi possível obter val_precision para trial {self.trial_count}: {e}")
                trial_data['best_val_precision'] = 0

            try:
                accuracy_history = trial.metrics.get_history('val_accuracy')
                if accuracy_history:
                    trial_data['best_val_accuracy'] = max(accuracy_history)
                    trial_data['accuracy_history'] = accuracy_history
                else:
                    LOGGER.debug(f"Trial {self.trial_count}: val_accuracy history está vazio")
            except (ValueError, KeyError) as e:
                LOGGER.debug(f"Trial {self.trial_count}: Não foi possível obter val_accuracy - {e}")

            try:
                auc_history = trial.metrics.get_history('val_AUC')
                if auc_history:
                    trial_data['best_val_auc'] = max(auc_history)
                    trial_data['auc_history'] = auc_history
                else:
                    LOGGER.debug(f"Trial {self.trial_count}: val_AUC history está vazio")
            except (ValueError, KeyError) as e:
                LOGGER.debug(f"Trial {self.trial_count}: Não foi possível obter val_AUC - {e}")

            try:
                loss_history = trial.metrics.get_history('val_loss')
                if loss_history:
                    trial_data['best_val_loss'] = min(loss_history)
                    trial_data['loss_history'] = loss_history
                else:
                    LOGGER.debug(f"Trial {self.trial_count}: val_loss history está vazio")
            except (ValueError, KeyError) as e:
                LOGGER.debug(f"Trial {self.trial_count}: Não foi possível obter val_loss - {e}")

        self.trials_data.append(trial_data)

        LOGGER.info(f"\nProgresso: {self.trial_count}/{self.total_trials} trials completados "
                   f"({100*self.trial_count/self.total_trials:.1f}%)")
        LOGGER.info("=" * 80 + "\n")

        return result

    def save_trials_data(self, output_dir):
        if not self.trials_data:
            LOGGER.warning("Nenhum dado de trial para salvar.")
            return

        os.makedirs(output_dir, exist_ok=True)

        trials_summary = []
        for trial_data in self.trials_data:
            summary = {
                'trial_id': trial_data['trial_id'],
                'trial_number': trial_data['trial_number'],
                'status': trial_data['status'],
                'best_val_precision': trial_data.get('best_val_precision', 0),
                'best_val_accuracy': trial_data.get('best_val_accuracy', 0),
                'best_val_auc': trial_data.get('best_val_auc', 0),
                'best_val_loss': trial_data.get('best_val_loss', 0),
            }

            hp = trial_data['hyperparameters']
            summary['num_layers'] = hp.get('num_layers', 0)
            summary['optimizer'] = hp.get('optimizer', 'N/A')
            summary['learning_rate'] = hp.get('learning_rate', 0)

            for i in range(summary['num_layers']):
                summary[f'layer_{i+1}_units'] = hp.get(f'units_layer_{i}', 0)
                summary[f'layer_{i+1}_activation'] = hp.get(f'activation_{i}', 'N/A')
                summary[f'layer_{i+1}_dropout'] = hp.get(f'dropout_{i}', 0)
                summary[f'layer_{i+1}_batch_norm'] = hp.get(f'batch_norm_{i}', False)
                summary[f'layer_{i+1}_l2_reg'] = hp.get(f'l2_reg_{i}', 0)

            trials_summary.append(summary)

        df_summary = pd.DataFrame(trials_summary)
        csv_path = os.path.join(output_dir, 'trials_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        LOGGER.info(f"Resumo dos trials salvo em: {csv_path}")

        json_path = os.path.join(output_dir, 'trials_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(self.trials_data, f, indent=2)
        LOGGER.info(f"Dados detalhados dos trials salvos em: {json_path}")

        best_trial = max(self.trials_data, key=lambda x: x.get('best_val_precision', 0))
        best_config_path = os.path.join(output_dir, 'best_trial_config.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'trial_id': best_trial['trial_id'],
                'trial_number': best_trial['trial_number'],
                'best_val_precision': best_trial.get('best_val_precision', 0),
                'hyperparameters': best_trial['hyperparameters']
            }, f, indent=2)
        LOGGER.info(f"Melhor configuração salva em: {best_config_path}")

        return df_summary


def build_tunable_mlp(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('input_dim'),)))

    num_layers = hp.Int('num_layers', min_value=2, max_value=5, step=1)

    for i in range(num_layers):
        units = hp.Int(f'units_layer_{i}', min_value=16, max_value=128, step=16)

        activation = hp.Choice(f'activation_{i}', values=['relu', 'elu', 'selu'])

        l2_reg = hp.Float(f'l2_reg_{i}', min_value=1e-4, max_value=1e-1, sampling='log')

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        use_batch_norm = hp.Boolean(f'batch_norm_{i}')
        if use_batch_norm:
            model.add(BatchNormalization())

        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.6, step=0.1)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', Precision(name='precision'), Recall(name='recall')]
    )

    return model


def tune_mlp_hyperparameters(x_train, y_train, x_val, y_val, max_trials=50, executions_per_trial=2):
    LOGGER.info("Iniciando busca de hiperparâmetros para MLP...")

    input_dim = x_train.shape[1]

    def model_builder(hp):
        hp.Fixed('input_dim', value=input_dim)
        return build_tunable_mlp(hp)

    tuner = CustomTuner(
        model_builder,
        objective=kt.Objective('val_precision', direction='max'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=os.path.join(RESULTS_DIR, 'tuning'),
        project_name='mlp_precision_optimization',
        overwrite=False
    )

    LOGGER.info(f"Busca configurada: {max_trials} trials, {executions_per_trial} executions per trial")
    LOGGER.info("Objetivo: Maximizar val_precision")

    class CustomEarlyStopping(keras.callbacks.EarlyStopping):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            if self.stopped_epoch > 0 and epoch == self.stopped_epoch:
                LOGGER.info(f"    🛑 EarlyStopping acionado na época {epoch+1}")
                LOGGER.info(f"       val_precision não melhorou por {self.patience} épocas")
                LOGGER.info(f"       Melhor val_precision: {logs.get('val_precision', 0):.4f}")

    early_stop = CustomEarlyStopping(
        monitor='val_precision',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=0
    )

    class EpochProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if (epoch + 1) % 5 == 0:
                LOGGER.info(
                    f"    Época {epoch+1}: "
                    f"loss={logs.get('loss', 0):.4f}, "
                    f"val_loss={logs.get('val_loss', 0):.4f}, "
                    f"val_precision={logs.get('val_precision', 0):.4f}, "
                    f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
                )

    epoch_progress = EpochProgressCallback()

    LOGGER.info("Iniciando busca (logs de progresso a cada 5 épocas)...")

    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop, epoch_progress],
        verbose=0
    )

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("SALVANDO DADOS DOS TRIALS")
    LOGGER.info("=" * 80)

    tuning_results_dir = os.path.join(RESULTS_DIR, 'tuning', 'results')
    df_trials = tuner.save_trials_data(tuning_results_dir)

    if df_trials is not None:
        LOGGER.info(f"\nTotal de trials executados: {len(df_trials)}")
        LOGGER.info(f"Melhor precisão alcançada: {df_trials['best_val_precision'].max():.4f}")
        LOGGER.info(f"Precisão média: {df_trials['best_val_precision'].mean():.4f}")
        LOGGER.info(f"Precisão mediana: {df_trials['best_val_precision'].median():.4f}")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("MELHORES HIPERPARÂMETROS ENCONTRADOS:")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Número de camadas: {best_hps.get('num_layers')}")

    for i in range(best_hps.get('num_layers')):
        LOGGER.info(f"\nCamada {i+1}:")
        LOGGER.info(f"  - Neurônios: {best_hps.get(f'units_layer_{i}')}")
        LOGGER.info(f"  - Ativação: {best_hps.get(f'activation_{i}')}")
        LOGGER.info(f"  - L2 Regularization: {best_hps.get(f'l2_reg_{i}'):.6f}")
        LOGGER.info(f"  - Batch Normalization: {best_hps.get(f'batch_norm_{i}')}")
        LOGGER.info(f"  - Dropout: {best_hps.get(f'dropout_{i}'):.2f}")

    LOGGER.info(f"\nOtimizador: {best_hps.get('optimizer')}")
    LOGGER.info(f"Learning Rate: {best_hps.get('learning_rate'):.6f}")
    LOGGER.info("=" * 80)

    best_model = tuner.get_best_models(num_models=1)[0]

    best_model_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Tuned_best.keras")
    best_model.save(best_model_path)
    LOGGER.info(f"Melhor modelo salvo em: {best_model_path}")

    results = tuner.results_summary(num_trials=5)

    return best_model, best_hps, tuner


def create_model_from_best_hps(best_hps, input_shape):
    LOGGER.info("Criando modelo MLP com melhores hiperparâmetros...")

    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = best_hps.get('num_layers')

    for i in range(num_layers):
        units = best_hps.get(f'units_layer_{i}')
        activation = best_hps.get(f'activation_{i}')
        l2_reg = best_hps.get(f'l2_reg_{i}')

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        if best_hps.get(f'batch_norm_{i}'):
            model.add(BatchNormalization())

        dropout_rate = best_hps.get(f'dropout_{i}')
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = best_hps.get('optimizer')
    learning_rate = best_hps.get('learning_rate')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', Precision(name='precision'), Recall(name='recall')]
    )

    return model


def load_and_create_model_from_config(config_path):
    LOGGER.info(f"Carregando configuração de: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    hyperparameters = config['hyperparameters']

    class HyperparametersDict:
        def __init__(self, values_dict):
            self.values_dict = values_dict

        def get(self, key):
            return self.values_dict.get(key)

    best_hps = HyperparametersDict(hyperparameters)
    input_shape = (hyperparameters.get('input_dim'),)

    model = create_model_from_best_hps(best_hps, input_shape)

    LOGGER.info(f"Modelo recriado a partir da configuração do trial {config['trial_number']}")
    LOGGER.info(f"Precisão alcançada originalmente: {config['best_val_precision']:.4f}")

    return model, hyperparameters


def analyze_tuning_results(tuner, top_n=10):
    LOGGER.info(f"\nAnalisando top {top_n} modelos...")

    best_trials = tuner.oracle.get_best_trials(num_trials=top_n)

    results = []
    for i, trial in enumerate(best_trials):
        metrics = trial.metrics.get_history('val_precision')
        if metrics:
            best_val_precision = max(metrics)
            results.append({
                'rank': i + 1,
                'trial_id': trial.trial_id,
                'val_precision': best_val_precision,
                'hyperparameters': trial.hyperparameters.values
            })

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"TOP {top_n} CONFIGURAÇÕES POR PRECISÃO:")
    LOGGER.info("=" * 80)

    for result in results:
        LOGGER.info(f"\nRank #{result['rank']} - Trial {result['trial_id']}")
        LOGGER.info(f"Val Precision: {result['val_precision']:.4f}")
        LOGGER.info(f"Config: {result['hyperparameters']}")

    return results
