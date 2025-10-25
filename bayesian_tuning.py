import os
import json
import pandas as pd
import numpy as np
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.metrics import Precision, Recall, AUC
from config import RESULTS_DIR, LOGGER


class BayesianTuner(kt.BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial_count = 0
        self.best_score = 0
        self.total_trials = kwargs.get('max_trials', 50)
        self.trials_data = []
        self.start_time = None

    def on_search_begin(self):
        import time
        self.start_time = time.time()
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("INICIANDO OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Total de trials: {self.total_trials}")
        LOGGER.info(f"Algoritmo: Bayesian Optimization")
        LOGGER.info(f"Objetivo: Maximizar val_precision")
        LOGGER.info("=" * 80 + "\n")

    def run_trial(self, trial, *args, **kwargs):
        self.trial_count += 1
        import time
        trial_start = time.time()

        LOGGER.info("=" * 80)
        LOGGER.info(f"TRIAL {self.trial_count}/{self.total_trials}")
        LOGGER.info("=" * 80)

        hp = trial.hyperparameters
        self._log_trial_config(hp)

        try:
            result = super().run_trial(trial, *args, **kwargs)
        except Exception as e:
            LOGGER.error(f"❌ Erro durante trial {self.trial_count}: {e}")
            import traceback
            LOGGER.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        trial_duration = time.time() - trial_start
        self._process_trial_results(trial, hp, trial_duration)

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

    def _process_trial_results(self, trial, hp, duration):
        import time

        trial_data = {
            'trial_id': trial.trial_id,
            'trial_number': self.trial_count,
            'status': trial.status,
            'duration_seconds': duration,
            'hyperparameters': dict(hp.values)
        }

        if hasattr(trial, 'metrics') and trial.metrics:
            metrics_collected = False

            try:
                precision_history = trial.metrics.get_history('val_precision')
                if precision_history:
                    best_precision = max(precision_history)
                    trial_data['best_val_precision'] = best_precision
                    trial_data['final_val_precision'] = precision_history[-1]
                    trial_data['precision_history'] = precision_history
                    metrics_collected = True

                    LOGGER.info(f"\n✓ Trial {self.trial_count} concluído em {duration:.1f}s")
                    LOGGER.info(f"  Melhor val_precision: {best_precision:.4f}")

                    if best_precision > self.best_score:
                        self.best_score = best_precision
                        improvement = ((best_precision - self.best_score) / self.best_score * 100) if self.best_score > 0 else 0
                        LOGGER.info(f"  ★★★ NOVO RECORDE! ★★★")
                        LOGGER.info(f"  Melhor score global: {self.best_score:.4f}")
                    else:
                        gap = self.best_score - best_precision
                        LOGGER.info(f"  Score global: {self.best_score:.4f} (gap: {gap:.4f})")
                else:
                    LOGGER.warning(f"Trial {self.trial_count}: val_precision history vazio")
                    trial_data['best_val_precision'] = 0
            except (ValueError, KeyError) as e:
                LOGGER.warning(f"Trial {self.trial_count}: Erro ao obter val_precision - {e}")
                trial_data['best_val_precision'] = 0

            for metric_name, metric_key in [
                ('accuracy', 'val_accuracy'),
                ('auc', 'val_auc'),
                ('recall', 'val_recall'),
                ('loss', 'val_loss')
            ]:
                try:
                    history = trial.metrics.get_history(metric_key)
                    if history:
                        if metric_name == 'loss':
                            trial_data[f'best_{metric_key}'] = min(history)
                        else:
                            trial_data[f'best_{metric_key}'] = max(history)
                        trial_data[f'{metric_key}_history'] = history
                except (ValueError, KeyError) as e:
                    LOGGER.debug(f"Trial {self.trial_count}: Métrica {metric_key} não disponível")

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

            if 'precision_history' in trial_data and trial_data['precision_history']:
                history_records = []

                precision_hist = trial_data.get('precision_history', [])
                accuracy_hist = trial_data.get('accuracy_history', [])
                auc_hist = trial_data.get('auc_history', [])
                recall_hist = trial_data.get('recall_history', [])
                loss_hist = trial_data.get('loss_history', [])

                max_epochs = max(
                    len(precision_hist) if precision_hist else 0,
                    len(accuracy_hist) if accuracy_hist else 0,
                    len(auc_hist) if auc_hist else 0,
                    len(recall_hist) if recall_hist else 0,
                    len(loss_hist) if loss_hist else 0
                )

                for epoch in range(max_epochs):
                    record = {
                        'trial_number': trial_num,
                        'trial_id': trial_data['trial_id'],
                        'epoch': epoch + 1,
                        'val_precision': precision_hist[epoch] if epoch < len(precision_hist) else None,
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

        all_histories = []
        for trial_data in self.trials_data:
            trial_num = trial_data['trial_number']

            precision_hist = trial_data.get('precision_history', [])
            accuracy_hist = trial_data.get('accuracy_history', [])
            auc_hist = trial_data.get('auc_history', [])
            recall_hist = trial_data.get('recall_history', [])
            loss_hist = trial_data.get('loss_history', [])

            max_epochs = max(
                len(precision_hist) if precision_hist else 0,
                len(accuracy_hist) if accuracy_hist else 0,
                len(auc_hist) if auc_hist else 0,
                len(recall_hist) if recall_hist else 0,
                len(loss_hist) if loss_hist else 0
            )

            for epoch in range(max_epochs):
                record = {
                    'trial_number': trial_num,
                    'trial_id': trial_data['trial_id'],
                    'epoch': epoch + 1,
                    'val_precision': precision_hist[epoch] if epoch < len(precision_hist) else None,
                    'val_accuracy': accuracy_hist[epoch] if epoch < len(accuracy_hist) else None,
                    'val_auc': auc_hist[epoch] if epoch < len(auc_hist) else None,
                    'val_recall': recall_hist[epoch] if epoch < len(recall_hist) else None,
                    'val_loss': loss_hist[epoch] if epoch < len(loss_hist) else None,
                }
                all_histories.append(record)

        if all_histories:
            df_all_histories = pd.DataFrame(all_histories)
            all_histories_path = os.path.join(output_dir, 'all_trials_histories_consolidated.csv')
            df_all_histories.to_csv(all_histories_path, index=False)
            LOGGER.info(f"\n✓ Histórico consolidado de todos os trials: {all_histories_path}")
            LOGGER.info(f"  Total de registros: {len(all_histories)} (épocas × trials)")

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
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )

    return model


def bayesian_tune_mlp(x_train, y_train, x_val, y_val, max_trials=30, executions_per_trial=2):
    LOGGER.info("=" * 80)
    LOGGER.info("CONFIGURANDO OTIMIZAÇÃO BAYESIANA")
    LOGGER.info("=" * 80)

    input_dim = x_train.shape[1]

    def model_builder(hp):
        hp.Fixed('input_dim', value=input_dim)
        batch_size = hp.Choice('batch_size', values=[32, 64, 128])
        return build_bayesian_mlp(hp)

    tuner = BayesianTuner(
        model_builder,
        objective=kt.Objective('val_precision', direction='max'),
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
                current_val = logs.get('val_precision', 0)
                LOGGER.info(f"    🛑 EarlyStopping na época {epoch+1}")
                LOGGER.info(f"       val_precision: {current_val:.4f} (sem melhoria por {self.patience} épocas)")

    early_stop = CustomEarlyStopping(
        monitor='val_precision',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=0
    )

    class EpochProgressCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_val_precision = 0

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_precision = logs.get('val_precision', 0)

            if current_precision > self.best_val_precision:
                self.best_val_precision = current_precision
                if (epoch + 1) % 5 == 0:
                    LOGGER.info(
                        f"    Época {epoch+1}: "
                        f"loss={logs.get('loss', 0):.4f}, "
                        f"val_loss={logs.get('val_loss', 0):.4f}, "
                        f"val_precision={current_precision:.4f} ⬆, "
                        f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
                    )
            elif (epoch + 1) % 10 == 0:
                LOGGER.info(
                    f"    Época {epoch+1}: "
                    f"val_precision={current_precision:.4f}, "
                    f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
                )

    epoch_progress = EpochProgressCallback()

    class BatchSizeCallback(keras.callbacks.Callback):
        def __init__(self, batch_size):
            super().__init__()
            self.batch_size = batch_size

        def on_train_begin(self, logs=None):
            LOGGER.info(f"    Iniciando treinamento com batch_size={self.batch_size}")

    LOGGER.info("🚀 Iniciando busca bayesiana...\n")

    tuner.on_search_begin()

    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop, epoch_progress],
        verbose=0
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
        LOGGER.info(f"    • Neurônios: {best_hps.get(f'units_layer_{i}')}")
        LOGGER.info(f"    • Ativação: {best_hps.get(f'activation_{i}')}")
        LOGGER.info(f"    • L2 Reg: {best_hps.get(f'l2_reg_{i}'):.6f}")
        LOGGER.info(f"    • Batch Norm: {best_hps.get(f'batch_norm_{i}')}")
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
            Precision(name='precision'),
            Recall(name='recall')
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
