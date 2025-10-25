import time
from config import LOGGER


class ProgressCallback:
    def __init__(self, max_trials, trial_type="MLP"):
        self.max_trials = max_trials
        self.trial_type = trial_type
        self.start_time = time.time()
        self.trial_times = []
        self.completed_trials = 0

    def on_trial_begin(self, trial):
        self.trial_start_time = time.time()
        self.completed_trials += 1

    def on_trial_end(self, trial):
        trial_time = time.time() - self.trial_start_time
        self.trial_times.append(trial_time)

        if hasattr(trial, 'score') and trial.score is not None:
            LOGGER.info(f"Trial completado em {self._format_time(trial_time)}")
            LOGGER.info(f"Score: {trial.score:.4f}")
        else:
            LOGGER.info(f"Trial completado em {self._format_time(trial_time)}")

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}min {secs}s"
        elif minutes > 0:
            return f"{minutes}min {secs}s"
        else:
            return f"{secs}s"

    def print_final_summary(self):
        total_time = time.time() - self.start_time
        avg_time = sum(self.trial_times) / len(self.trial_times) if self.trial_times else 0

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("RESUMO DA BUSCA DE HIPERPARÂMETROS")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Tipo: {self.trial_type}")
        LOGGER.info(f"Total de trials: {self.completed_trials}")
        LOGGER.info(f"Tempo total: {self._format_time(total_time)}")
        LOGGER.info(f"Tempo médio por trial: {self._format_time(avg_time)}")
        LOGGER.info(f"Tempo mínimo: {self._format_time(min(self.trial_times)) if self.trial_times else 0}")
        LOGGER.info(f"Tempo máximo: {self._format_time(max(self.trial_times)) if self.trial_times else 0}")
        LOGGER.info("=" * 80)


class KerasTunerProgressCallback:
    def __init__(self, max_trials, trial_type="MLP"):
        self.progress = ProgressCallback(max_trials, trial_type)

    def on_trial_begin(self, tuner, trial):
        self.progress.on_trial_begin(trial)

    def on_trial_end(self, tuner, trial):
        self.progress.on_trial_end(trial)
