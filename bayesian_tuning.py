import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import json
import pandas as pd
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.metrics import Precision, Recall, AUC, PrecisionAtRecall
from config import RESULTS_DIR, LOGGER, DEFAULT_TUNING_EPOCHS


class BayesianTuner(kt.BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_count = 0
        self.best_score = 0
        self.total_trials = kwargs.get('max_trials', 50)
        self.trials_data = []
        self.start_time = None
        self.current_history = None

    def on_search_begin(self):
        import time
        self.start_time = time.time()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        self.trial_count += 1
        import time
        trial_start = time.time()

        LOGGER.info("=" * 80)
        LOGGER.info(f"TRIAL {self.trial_count}/{self.total_trials}")
        LOGGER.info("=" * 80)

        hp = trial.hyperparameters
        self._log_trial_config(hp)

        class HistoryCapture(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.history_dict = {
                    'loss': [],
                    'accuracy': [],
                    'auc': [],
                    'precision': [],
                    'recall': [],
                    'pr_auc': [],
                    'val_loss': [],
                    'val_accuracy': [],
                    'val_auc': [],
                    'val_precision': [],
                    'val_recall': [],
                    'val_pr_auc': []
                }

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for key in list(self.history_dict.keys()):
                    if key in logs:
                        self.history_dict[key].append(float(logs[key]))

        history_callback = HistoryCapture()

        if 'callbacks' not in fit_kwargs:
            fit_kwargs['callbacks'] = []
        fit_kwargs['callbacks'].append(history_callback)

        try:
            result = super().run_trial(trial, *fit_args, **fit_kwargs)
            history = history_callback.history_dict

        except Exception as e:
            LOGGER.error(f"❌ Erro durante trial {self.trial_count}: {e}")
            import traceback
            LOGGER.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        trial_duration = time.time() - trial_start
        self._process_trial_results(trial, hp, trial_duration, history)

        return result

    def _log_trial_config(self, hp):
        LOGGER.info(f"Configuração:")

        num_layers = hp.values.get('num_layers', 0)
        LOGGER.info(f"  • Arquitetura: {num_layers} camadas")

        optimizer = hp.values.get('optimizer', 'N/A')
        lr = hp.values.get('learning_rate', 0)
        LOGGER.info(f"  • Otimizador: {optimizer} (lr={lr:.6f})")

        batch_size = hp.values.get('batch_size', 64)
        LOGGER.info(f"  • Batch Size: {batch_size}")

        for i in range(num_layers):
            units = hp.values.get(f'units_layer_{i}', 'N/A')
            activation = hp.values.get(f'activation_{i}', 'N/A')
            dropout = hp.values.get(f'dropout_{i}', 0)
            batch_norm = hp.values.get(f'batch_norm_{i}', False)
            l2_reg = hp.values.get(f'l2_reg_{i}', 0)

            LOGGER.info(f"  • Camada {i+1}: {units} neurônios | "
                       f"ativação={activation} | "
                       f"dropout={dropout:.2f} | "
                       f"batch_norm={batch_norm} | "
                       f"l2={l2_reg:.6f}")

    def _process_trial_results(self, trial, hp, duration, history=None):
        import time

        trial_data = {
            'trial_id': trial.trial_id,
            'trial_number': self.trial_count,
            'status': trial.status,
            'duration_seconds': duration,
            'hyperparameters': dict(hp.values)
        }

        try:
            # Preferir histórico capturado; fallback para trial.metrics
            precision_hist = (history or {}).get('val_precision', []) if history else []
            pr_auc_hist = (history or {}).get('val_pr_auc', []) if history else []

            if not precision_hist:
                try:
                    precision_hist = trial.metrics.get_history('val_precision') or []
                except Exception:
                    precision_hist = []
            if not pr_auc_hist:
                try:
                    pr_auc_hist = trial.metrics.get_history('val_pr_auc') or []
                except Exception:
                    pr_auc_hist = []

            if precision_hist:
                best_precision = max(precision_hist)
                trial_data['best_val_precision'] = best_precision
                trial_data['final_val_precision'] = precision_hist[-1]
                trial_data['precision_history'] = precision_hist
            else:
                LOGGER.warning(f"Trial {self.trial_count}: val_precision history vazio")
                trial_data['best_val_precision'] = 0

            if pr_auc_hist:
                best_pr_auc = max(pr_auc_hist)
                trial_data['best_val_pr_auc'] = best_pr_auc
                trial_data['final_val_pr_auc'] = pr_auc_hist[-1]
                trial_data['pr_auc_history'] = pr_auc_hist
            else:
                trial_data['best_val_pr_auc'] = 0

            # Outras métricas
            for metric_name, metric_key in [
                ('accuracy', 'val_accuracy'),
                ('auc', 'val_auc'),
                ('recall', 'val_recall'),
                ('loss', 'val_loss')
            ]:
                hist = (history or {}).get(metric_key, [])
                if not hist:
                    try:
                        hist = trial.metrics.get_history(metric_key) or []
                    except Exception:
                        hist = []
                if hist:
                    if metric_name == 'loss':
                        trial_data[f'best_{metric_key}'] = min(hist)
                    else:
                        trial_data[f'best_{metric_key}'] = max(hist)
                    trial_data[f'{metric_key}_history'] = hist

            # Score de comparação permanece precisão (compatibilidade), mas registramos PR AUC
            best_precision = trial_data.get('best_val_precision', 0)
            if best_precision > self.best_score:
                self.best_score = best_precision

            LOGGER.info(f"\n✓ Trial {self.trial_count} concluído em {duration:.1f}s")
            LOGGER.info(f"  Melhor val_precision: {trial_data.get('best_val_precision', 0):.4f}")
            LOGGER.info(f"  Melhor val_pr_auc: {trial_data.get('best_val_pr_auc', 0):.4f}")

        except Exception as e:
            LOGGER.warning(f"Trial {self.trial_count}: Erro ao processar métricas - {e}")
            trial_data['best_val_precision'] = 0
            trial_data['best_val_pr_auc'] = 0

        self.trials_data.append(trial_data)

        progress_pct = (self.trial_count / self.total_trials) * 100
        remaining = self.total_trials - self.trial_count

        LOGGER.info(f"\n📊 Progresso: {self.trial_count}/{self.total_trials} trials ({progress_pct:.1f}%)")
        LOGGER.info(f"   Restam: {remaining} trials")

        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / self.trial_count
            estimated_remaining = avg_time * remaining
            LOGGER.info(f"   Tempo estimado restante: {estimated_remaining/60:.1f} minutos")

        LOGGER.info("=" * 80 + "\n")

    def save_trials_data(self, output_dir):
        if not self.trials_data:
            LOGGER.warning("Nenhum dado de trial para salvar.")
            return None

        os.makedirs(output_dir, exist_ok=True)

        trials_summary = []
        for trial_data in self.trials_data:
            summary = {
                'trial_id': trial_data['trial_id'],
                'trial_number': trial_data['trial_number'],
                'status': trial_data['status'],
                'duration_seconds': trial_data.get('duration_seconds', 0),
                'best_val_precision': trial_data.get('best_val_precision', 0),
                'best_val_pr_auc': trial_data.get('best_val_pr_auc', 0),
                'best_val_accuracy': trial_data.get('best_val_accuracy', 0),
                'best_val_auc': trial_data.get('best_val_auc', 0),
                'best_val_recall': trial_data.get('best_val_recall', 0),
                'best_val_loss': trial_data.get('best_val_loss', 0),
            }

            hp = trial_data['hyperparameters']
            summary['num_layers'] = hp.get('num_layers', 0)
            summary['optimizer'] = hp.get('optimizer', 'N/A')
            summary['learning_rate'] = hp.get('learning_rate', 0)
            summary['batch_size'] = hp.get('batch_size', 64)

            for i in range(summary['num_layers']):
                summary[f'layer_{i+1}_units'] = hp.get(f'units_layer_{i}', 0)
                summary[f'layer_{i+1}_activation'] = hp.get(f'activation_{i}', 'N/A')
                summary[f'layer_{i+1}_dropout'] = hp.get(f'dropout_{i}', 0)
                summary[f'layer_{i+1}_batch_norm'] = hp.get(f'batch_norm_{i}', False)
                summary[f'layer_{i+1}_l2_reg'] = hp.get(f'l2_reg_{i}', 0)

            trials_summary.append(summary)

        df_summary = pd.DataFrame(trials_summary)
        csv_path = os.path.join(output_dir, 'bayesian_trials_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        LOGGER.info(f"✓ Resumo dos trials salvo: {csv_path}")

        history_dir = os.path.join(output_dir, 'trial_histories')
        os.makedirs(history_dir, exist_ok=True)

        LOGGER.info(f"\n💾 Salvando históricos individuais de cada trial...")
        for trial_data in self.trials_data:
            trial_num = trial_data['trial_number']

            precision_hist = trial_data.get('precision_history', [])
            pr_auc_hist = trial_data.get('pr_auc_history', [])
            accuracy_hist = trial_data.get('val_accuracy_history', trial_data.get('accuracy_history', []))
            auc_hist = trial_data.get('val_auc_history', trial_data.get('auc_history', []))
            recall_hist = trial_data.get('val_recall_history', trial_data.get('recall_history', []))
            loss_hist = trial_data.get('val_loss_history', trial_data.get('loss_history', []))

            max_epochs = max(
                len(precision_hist), len(pr_auc_hist), len(accuracy_hist), len(auc_hist), len(recall_hist), len(loss_hist)
            )

            history_records = []
            for epoch in range(max_epochs):
                record = {
                    'trial_number': trial_num,
                    'trial_id': trial_data['trial_id'],
                    'epoch': epoch + 1,
                    'val_precision': precision_hist[epoch] if epoch < len(precision_hist) else None,
                    'val_pr_auc': pr_auc_hist[epoch] if epoch < len(pr_auc_hist) else None,
                    'val_accuracy': accuracy_hist[epoch] if epoch < len(accuracy_hist) else None,
                    'val_auc': auc_hist[epoch] if epoch < len(auc_hist) else None,
                    'val_recall': recall_hist[epoch] if epoch < len(recall_hist) else None,
                    'val_loss': loss_hist[epoch] if epoch < len(loss_hist) else None,
                }
                history_records.append(record)

            if history_records:
                df_history = pd.DataFrame(history_records)
                history_csv_path = os.path.join(history_dir, f'trial_{trial_num:03d}_history.csv')
                df_history.to_csv(history_csv_path, index=False)
                LOGGER.info(f"  ✓ Trial {trial_num}: {len(history_records)} épocas salvas")

        json_path = os.path.join(output_dir, 'bayesian_trials_detailed.json')
        with open(json_path, 'w') as f:
            json.dump(self.trials_data, f, indent=2)
        LOGGER.info(f"✓ Dados detalhados salvos: {json_path}")

        best_trial = max(self.trials_data, key=lambda x: x.get('best_val_precision', 0))
        best_config_path = os.path.join(output_dir, 'bayesian_best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'trial_id': best_trial['trial_id'],
                'trial_number': best_trial['trial_number'],
                'best_val_precision': best_trial.get('best_val_precision', 0),
                'best_val_pr_auc': best_trial.get('best_val_pr_auc', 0),
                'duration_seconds': best_trial.get('duration_seconds', 0),
                'hyperparameters': best_trial['hyperparameters']
            }, f, indent=2)
        LOGGER.info(f"✓ Melhor configuração salva: {best_config_path}")

        self._generate_analysis(df_summary, output_dir)

        return df_summary

    def _generate_analysis(self, df_summary, output_dir):
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("ANÁLISE DOS RESULTADOS")
        LOGGER.info("=" * 80)

        LOGGER.info(f"\n📈 Estatísticas de Precisão:")
        LOGGER.info(f"  • Melhor: {df_summary['best_val_precision'].max():.4f}")
        LOGGER.info(f"  • Média: {df_summary['best_val_precision'].mean():.4f}")
        LOGGER.info(f"  • Mediana: {df_summary['best_val_precision'].median():.4f}")
        LOGGER.info(f"  • Desvio Padrão: {df_summary['best_val_precision'].std():.4f}")
        LOGGER.info(f"  • Pior: {df_summary['best_val_precision'].min():.4f}")
        if 'best_val_pr_auc' in df_summary.columns:
            LOGGER.info(f"\n📈 Estatísticas de PR AUC:")
            LOGGER.info(f"  • Melhor: {df_summary['best_val_pr_auc'].max():.4f}")
            LOGGER.info(f"  • Média: {df_summary['best_val_pr_auc'].mean():.4f}")
            LOGGER.info(f"  • Mediana: {df_summary['best_val_pr_auc'].median():.4f}")
            LOGGER.info(f"  • Desvio Padrão: {df_summary['best_val_pr_auc'].std():.4f}")

        LOGGER.info(f"\n⏱️  Tempo de Execução:")
        total_time = df_summary['duration_seconds'].sum()
        LOGGER.info(f"  • Total: {total_time/60:.1f} minutos")
        LOGGER.info(f"  • Média por trial: {df_summary['duration_seconds'].mean():.1f} segundos")

        top_configs = df_summary.nlargest(5, 'best_val_precision')
        LOGGER.info(f"\n🏆 Top 5 Configurações:")
        for idx, row in top_configs.iterrows():
            LOGGER.info(f"  #{row['trial_number']}: precision={row['best_val_precision']:.4f}, "
                       f"layers={row['num_layers']}, optimizer={row['optimizer']}")


def build_bayesian_mlp(hp):
    model = Sequential()
    model.add(Input(shape=(hp.get('input_dim'),)))

    num_layers = hp.Int('num_layers', min_value=2, max_value=5, step=1)

    for i in range(num_layers):
        units = hp.Int(f'units_layer_{i}', min_value=32, max_value=256, step=32)

        activation = hp.Choice(f'activation_{i}', values=['relu', 'elu', 'selu', 'swish'])

        l2_reg = hp.Float(f'l2_reg_{i}', min_value=1e-5, max_value=1e-2, sampling='log')

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        use_batch_norm = hp.Boolean(f'batch_norm_{i}')
        if use_batch_norm:
            model.add(BatchNormalization())

        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.6, step=0.1)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            AUC(curve='PR', name='pr_auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            PrecisionAtRecall(0.80, name='precision_at_recall_80')
        ]
    )

    return model


class HyperModelWithBatchSize(kt.HyperModel):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def build(self, hp):
        hp.Fixed('input_dim', value=self.input_dim)
        hp.Choice('batch_size', values=[32, 64, 128])
        return build_bayesian_mlp(hp)

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.get('batch_size')
        kwargs['batch_size'] = batch_size
        return model.fit(*args, **kwargs)


def bayesian_tune_mlp(x_train, y_train, x_val, y_val, max_trials=30, executions_per_trial=2, progress_callback=None, epochs=None):
    if epochs is None:
        epochs = DEFAULT_TUNING_EPOCHS

    LOGGER.info("=" * 80)
    LOGGER.info("CONFIGURANDO OTIMIZAÇÃO BAYESIANA")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Épocas por trial: {epochs}")

    input_dim = x_train.shape[1]

    hypermodel = HyperModelWithBatchSize(input_dim)

    tuner = BayesianTuner(
        hypermodel,
        objective=kt.Objective('val_pr_auc', direction='max'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=os.path.join(RESULTS_DIR, 'tuning'),
        project_name='bayesian_mlp_optimization',
        overwrite=False,
        num_initial_points=5,
        alpha=1e-4,
        beta=2.6
    )

    LOGGER.info(f"✓ Configuração completa:")
    LOGGER.info(f"  • Max trials: {max_trials}")
    LOGGER.info(f"  • Executions per trial: {executions_per_trial}")
    LOGGER.info(f"  • Initial random points: 5")
    LOGGER.info(f"  • Acquisition function: Expected Improvement")
    LOGGER.info("=" * 80 + "\n")

    class CustomEarlyStopping(keras.callbacks.EarlyStopping):
        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            if self.stopped_epoch > 0 and epoch == self.stopped_epoch:
                current_val = logs.get('val_pr_auc', 0)
                LOGGER.info(f"    🛑 EarlyStopping na época {epoch+1}")
                LOGGER.info(f"       val_pr_auc: {current_val:.4f} (sem melhoria por {self.patience} épocas)")

    early_stop = CustomEarlyStopping(
        monitor='val_pr_auc',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    class EpochProgressCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_val_precision = 0
            self.best_val_pr_auc = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_precision = logs.get('val_precision', 0)
            current_pr_auc = logs.get('val_pr_auc', 0)
            improved = False
            if current_precision > self.best_val_precision:
                self.best_val_precision = current_precision
                improved = True
            if current_pr_auc > self.best_val_pr_auc:
                self.best_val_pr_auc = current_pr_auc
                improved = True
            if improved and (epoch + 1) % 5 == 0:
                LOGGER.info(
                    f"    Época {epoch+1}: "
                    f"loss={logs.get('loss', 0):.4f}, "
                    f"val_loss={logs.get('val_loss', 0):.4f}, "
                    f"val_precision={current_precision:.4f} ⬆, "
                    f"val_pr_auc={current_pr_auc:.4f} ⬆, "
                    f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
                )
            elif (epoch + 1) % 10 == 0:
                LOGGER.info(
                    f"    Época {epoch+1}: "
                    f"val_precision={current_precision:.4f}, val_pr_auc={current_pr_auc:.4f}, "
                    f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
                )

    epoch_progress = EpochProgressCallback()

    LOGGER.info("🚀 Iniciando busca bayesiana...\n")

    if progress_callback:
        original_run_trial = tuner.run_trial

        def wrapped_run_trial(trial, *args, **kwargs):
            progress_callback.on_trial_begin(trial)
            result = original_run_trial(trial, *args, **kwargs)
            progress_callback.on_trial_end(trial)
            return result

        tuner.run_trial = wrapped_run_trial

    tuner.on_search_begin()

    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=[early_stop, epoch_progress],
        verbose=1
    )

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("SALVANDO RESULTADOS")
    LOGGER.info("=" * 80)

    tuning_results_dir = os.path.join(RESULTS_DIR, 'tuning', 'bayesian_results')
    df_trials = tuner.save_trials_data(tuning_results_dir)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("🏆 MELHOR CONFIGURAÇÃO ENCONTRADA")
    LOGGER.info("=" * 80)

    num_layers = best_hps.get('num_layers')
    LOGGER.info(f"\n📐 Arquitetura: {num_layers} camadas")

    for i in range(num_layers):
        LOGGER.info(f"\n  Camada {i+1}:")
        LOGGER.info(f"    • Neurônios: {best_hps.get(f'units_layer_{i}')} ")
        LOGGER.info(f"    • Ativação: {best_hps.get(f'activation_{i}')} ")
        LOGGER.info(f"    • L2 Reg: {best_hps.get(f'l2_reg_{i}'):.6f} ")
        LOGGER.info(f"    • Batch Norm: {best_hps.get(f'batch_norm_{i}')} ")
        LOGGER.info(f"    • Dropout: {best_hps.get(f'dropout_{i}'):.2f}")

    LOGGER.info(f"\n⚙️  Otimização:")
    LOGGER.info(f"    • Otimizador: {best_hps.get('optimizer')}")
    LOGGER.info(f"    • Learning Rate: {best_hps.get('learning_rate'):.6f}")
    batch_size_value = best_hps.get('batch_size') if 'batch_size' in best_hps.values else 64
    LOGGER.info(f"    • Batch Size: {batch_size_value}")
    LOGGER.info("=" * 80)

    best_model = tuner.get_best_models(num_models=1)[0]

    best_model_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Bayesian_best.keras")
    best_model.save(best_model_path)
    LOGGER.info(f"\n✓ Melhor modelo salvo: {best_model_path}")

    tuner.results_summary(num_trials=5)

    return best_model, best_hps, tuner


def create_model_from_bayesian_config(config_path):
    LOGGER.info(f"Carregando configuração bayesiana: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    hp = config['hyperparameters']

    model = Sequential()
    model.add(Input(shape=(hp.get('input_dim'),)))

    num_layers = hp.get('num_layers')

    for i in range(num_layers):
        units = hp.get(f'units_layer_{i}')
        activation = hp.get(f'activation_{i}')
        l2_reg = hp.get(f'l2_reg_{i}')

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        if hp.get(f'batch_norm_{i}'):
            model.add(BatchNormalization())

        dropout_rate = hp.get(f'dropout_{i}')
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = hp.get('optimizer')
    learning_rate = hp.get('learning_rate')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            AUC(curve='PR', name='pr_auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            PrecisionAtRecall(0.80, name='precision_at_recall_80')
        ]
    )

    LOGGER.info(f"✓ Modelo recriado do trial #{config['trial_number']}")
    LOGGER.info(f"  Precisão original: {config['best_val_precision']:.4f}")

    return model, hp


def compare_tuning_approaches(standard_results_path, bayesian_results_path):
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("COMPARAÇÃO: Tuning Padrão vs Bayesiano")
    LOGGER.info("=" * 80)

    df_standard = pd.read_csv(standard_results_path)
    df_bayesian = pd.read_csv(bayesian_results_path)

    LOGGER.info("\n📊 Resultados Padrão:")
    LOGGER.info(f"  • Melhor precisão: {df_standard['best_val_precision'].max():.4f}")
    LOGGER.info(f"  • Precisão média: {df_standard['best_val_precision'].mean():.4f}")
    LOGGER.info(f"  • Total de trials: {len(df_standard)}")

    LOGGER.info("\n📊 Resultados Bayesianos:")
    LOGGER.info(f"  • Melhor precisão: {df_bayesian['best_val_precision'].max():.4f}")
    LOGGER.info(f"  • Precisão média: {df_bayesian['best_val_precision'].mean():.4f}")
    LOGGER.info(f"  • Total de trials: {len(df_bayesian)}")

    improvement = df_bayesian['best_val_precision'].max() - df_standard['best_val_precision'].max()
    LOGGER.info(f"\n{'✓' if improvement > 0 else '✗'} Diferença: {improvement:+.4f}")

    return {
        'standard_best': df_standard['best_val_precision'].max(),
        'bayesian_best': df_bayesian['best_val_precision'].max(),
        'improvement': improvement
    }


def analyze_trial_selection(results_dir):
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ANÁLISE DETALHADA DA SELEÇÃO DE TRIALS")
    LOGGER.info("=" * 80)

    summary_path = os.path.join(results_dir, 'bayesian_trials_summary.csv')
    detailed_path = os.path.join(results_dir, 'bayesian_trials_detailed.json')
    best_config_path = os.path.join(results_dir, 'bayesian_best_config.json')

    if not os.path.exists(summary_path):
        LOGGER.error(f"Arquivo não encontrado: {summary_path}")
        return None

    df = pd.read_csv(summary_path)

    with open(best_config_path, 'r') as f:
        best_config = json.load(f)

    best_trial_num = best_config['trial_number']
    best_precision = best_config['best_val_precision']

    LOGGER.info(f"\n🏆 MODELO SELECIONADO:")
    LOGGER.info(f"  • Trial #{best_trial_num}")
    LOGGER.info(f"  • Precisão de Validação: {best_precision:.4f}")
    LOGGER.info(f"  • Critério de Seleção: Máxima val_precision")

    df_sorted = df.sort_values('best_val_precision', ascending=False)

    LOGGER.info(f"\n📊 RANKING DOS TRIALS (Top 10):")
    LOGGER.info(f"{'Rank':<6} {'Trial':<8} {'Precision':<12} {'Accuracy':<12} {'AUC':<12} {'Layers':<8} {'Optimizer':<12}")
    LOGGER.info("-" * 80)

    for rank, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        marker = "★" if row['trial_number'] == best_trial_num else " "
        LOGGER.info(
            f"{marker} {rank:<5} "
            f"#{int(row['trial_number']):<7} "
            f"{row['best_val_precision']:.4f}      "
            f"{row['best_val_accuracy']:.4f}      "
            f"{row['best_val_auc']:.4f}      "
            f"{int(row['num_layers']):<7} "
            f"{row['optimizer']:<12}"
        )

    LOGGER.info(f"\n📈 ESTATÍSTICAS COMPARATIVAS:")

    best_row = df[df['trial_number'] == best_trial_num].iloc[0]

    percentile_precision = (df['best_val_precision'] < best_precision).sum() / len(df) * 100
    LOGGER.info(f"  • Percentil do modelo selecionado: {percentile_precision:.1f}%")

    gap_to_second = best_precision - df_sorted.iloc[1]['best_val_precision']
    LOGGER.info(f"  • Diferença para o 2º lugar: {gap_to_second:.4f}")

    gap_to_median = best_precision - df['best_val_precision'].median()
    LOGGER.info(f"  • Diferença para a mediana: {gap_to_median:.4f}")

    LOGGER.info(f"\n🔍 CONFIGURAÇÃO DO MODELO SELECIONADO:")
    LOGGER.info(f"  • Arquitetura: {int(best_row['num_layers'])} camadas")
    LOGGER.info(f"  • Otimizador: {best_row['optimizer']}")
    LOGGER.info(f"  • Learning Rate: {best_row['learning_rate']:.6f}")
    LOGGER.info(f"  • Batch Size: {int(best_row['batch_size'])}")

    for i in range(int(best_row['num_layers'])):
        LOGGER.info(f"\n  Camada {i+1}:")
        LOGGER.info(f"    - Neurônios: {int(best_row[f'layer_{i+1}_units'])}")
        LOGGER.info(f"    - Ativação: {best_row[f'layer_{i+1}_activation']}")
        LOGGER.info(f"    - Dropout: {best_row[f'layer_{i+1}_dropout']:.2f}")
        LOGGER.info(f"    - Batch Norm: {best_row[f'layer_{i+1}_batch_norm']}")
        LOGGER.info(f"    - L2 Reg: {best_row[f'layer_{i+1}_l2_reg']:.6f}")

    return {
        'best_trial': best_trial_num,
        'best_precision': best_precision,
        'percentile': percentile_precision,
        'gap_to_second': gap_to_second,
        'ranking': df_sorted[['trial_number', 'best_val_precision', 'best_val_accuracy', 'best_val_auc']].head(10)
    }


def compare_all_trials(results_dir, metric='best_val_precision'):
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"COMPARAÇÃO ENTRE TODOS OS TRIALS - Métrica: {metric}")
    LOGGER.info("=" * 80)

    summary_path = os.path.join(results_dir, 'bayesian_trials_summary.csv')
    df = pd.read_csv(summary_path)

    LOGGER.info(f"\n📊 ESTATÍSTICAS GERAIS:")
    LOGGER.info(f"  • Total de trials: {len(df)}")
    LOGGER.info(f"  • Melhor {metric}: {df[metric].max():.4f}")
    LOGGER.info(f"  • Pior {metric}: {df[metric].min():.4f}")
    LOGGER.info(f"  • Média: {df[metric].mean():.4f}")
    LOGGER.info(f"  • Mediana: {df[metric].median():.4f}")
    LOGGER.info(f"  • Desvio Padrão: {df[metric].std():.4f}")
    LOGGER.info(f"  • Amplitude: {df[metric].max() - df[metric].min():.4f}")

    LOGGER.info(f"\n🔍 ANÁLISE POR HIPERPARÂMETROS:")

    LOGGER.info(f"\n  Número de Camadas:")
    for layers in sorted(df['num_layers'].unique()):
        subset = df[df['num_layers'] == layers]
        LOGGER.info(f"    {int(layers)} camadas: {len(subset)} trials, média={subset[metric].mean():.4f}, max={subset[metric].max():.4f}")

    LOGGER.info(f"\n  Otimizador:")
    for opt in df['optimizer'].unique():
        subset = df[df['optimizer'] == opt]
        LOGGER.info(f"    {opt}: {len(subset)} trials, média={subset[metric].mean():.4f}, max={subset[metric].max():.4f}")

    LOGGER.info(f"\n  Batch Size:")
    for bs in sorted(df['batch_size'].unique()):
        subset = df[df['batch_size'] == bs]
        LOGGER.info(f"    {int(bs)}: {len(subset)} trials, média={subset[metric].mean():.4f}, max={subset[metric].max():.4f}")

    LOGGER.info(f"\n⏱️  ANÁLISE TEMPORAL:")
    LOGGER.info(f"  • Tempo total: {df['duration_seconds'].sum()/60:.1f} minutos")
    LOGGER.info(f"  • Tempo médio por trial: {df['duration_seconds'].mean():.1f} segundos")
    LOGGER.info(f"  • Trial mais rápido: {df['duration_seconds'].min():.1f}s")
    LOGGER.info(f"  • Trial mais lento: {df['duration_seconds'].max():.1f}s")

    LOGGER.info(f"\n📈 CONVERGÊNCIA:")
    df_sorted = df.sort_values('trial_number')
    cumulative_best = df_sorted[metric].cummax()

    improvements = (cumulative_best.diff() > 0).sum()
    LOGGER.info(f"  • Número de melhorias: {improvements} trials")
    LOGGER.info(f"  • Taxa de melhoria: {improvements/len(df)*100:.1f}%")

    first_best_trial = df_sorted[df_sorted[metric] == df[metric].max()]['trial_number'].min()
    LOGGER.info(f"  • Melhor resultado encontrado no trial: #{int(first_best_trial)}")
    LOGGER.info(f"  • Trials até encontrar o melhor: {int(first_best_trial)} de {len(df)} ({first_best_trial/len(df)*100:.1f}%)")

    return df


def analyze_hyperparameter_correlations(results_dir):
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ANÁLISE DE CORRELAÇÕES ENTRE HIPERPARÂMETROS E DESEMPENHO")
    LOGGER.info("=" * 80)

    summary_path = os.path.join(results_dir, 'bayesian_trials_summary.csv')
    df = pd.read_csv(summary_path)

    numeric_cols = ['num_layers', 'learning_rate', 'batch_size',
                    'best_val_precision', 'best_val_accuracy', 'best_val_auc']

    correlations = df[numeric_cols].corr()['best_val_precision'].sort_values(ascending=False)

    LOGGER.info(f"\n📊 CORRELAÇÕES COM PRECISÃO DE VALIDAÇÃO:")
    for param, corr in correlations.items():
        if param != 'best_val_precision':
            strength = "forte" if abs(corr) > 0.5 else "moderada" if abs(corr) > 0.3 else "fraca"
            direction = "positiva" if corr > 0 else "negativa"
            LOGGER.info(f"  • {param}: {corr:+.3f} ({strength} {direction})")

    LOGGER.info(f"\n🎯 ANÁLISE POR FUNÇÃO DE ATIVAÇÃO:")
    for i in range(1, 6):
        col = f'layer_{i}_activation'
        if col in df.columns:
            activation_stats = df.groupby(col)['best_val_precision'].agg(['count', 'mean', 'max', 'std'])
            if not activation_stats.empty:
                LOGGER.info(f"\n  Camada {i}:")
                for act, row in activation_stats.iterrows():
                    if pd.notna(act):
                        LOGGER.info(f"    {act}: {int(row['count'])} trials, "
                                  f"média={row['mean']:.4f}, max={row['max']:.4f}, std={row['std']:.4f}")

    LOGGER.info(f"\n🔄 ANÁLISE DE DROPOUT:")
    for i in range(1, 6):
        col = f'layer_{i}_dropout'
        if col in df.columns and df[col].notna().any():
            dropout_corr = df[[col, 'best_val_precision']].corr().iloc[0, 1]
            LOGGER.info(f"  Camada {i} dropout vs precision: {dropout_corr:+.3f}")

    LOGGER.info(f"\n✨ ANÁLISE DE BATCH NORMALIZATION:")
    for i in range(1, 6):
        col = f'layer_{i}_batch_norm'
        if col in df.columns:
            with_bn = df[df[col] == True]['best_val_precision'].mean()
            without_bn = df[df[col] == False]['best_val_precision'].mean()
            if pd.notna(with_bn) and pd.notna(without_bn):
                diff = with_bn - without_bn
                LOGGER.info(f"  Camada {i}: Com BN={with_bn:.4f}, Sem BN={without_bn:.4f}, Diferença={diff:+.4f}")

    return correlations


def export_comparison_report(results_dir, output_file='trial_comparison_report.txt'):
    report_path = os.path.join(results_dir, output_file)

    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        analyze_trial_selection(results_dir)
        compare_all_trials(results_dir)
        analyze_hyperparameter_correlations(results_dir)

        report_content = captured_output.getvalue()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        sys.stdout = old_stdout
        LOGGER.info(f"\n✓ Relatório de comparação salvo: {report_path}")

        return report_path

    finally:
        sys.stdout = old_stdout
