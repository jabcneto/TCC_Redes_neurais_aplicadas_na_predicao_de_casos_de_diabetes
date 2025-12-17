"""
Módulo para rastreamento e logging de tempos de treinamento dos modelos.
"""

import json
import os
from datetime import datetime
from config import RESULTS_DIR, LOGGER


class TrainingTimeTracker:
    """Rastreia tempos de treinamento para todos os modelos."""
    
    def __init__(self):
        self.training_times = {}
        self.metrics_dir = os.path.join(RESULTS_DIR, "tempo_treinamento")
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def register_model_time(self, model_name: str, elapsed_time: float):
        """Registra o tempo de treinamento de um modelo."""
        self.training_times[model_name] = {
            'tempo_segundos': round(elapsed_time, 2),
            'tempo_minutos': round(elapsed_time / 60, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def log_training_summary(self):
        """Exibe um resumo formatado dos tempos de treinamento."""
        if not self.training_times:
            return
        
        LOGGER.info(f"\n{'='*70}")
        LOGGER.info("⏱️  RESUMO DE TEMPO DE TREINAMENTO - TODOS OS MODELOS")
        LOGGER.info(f"{'='*70}")
        
        # Ordenar por tempo decrescente
        sorted_times = sorted(
            self.training_times.items(),
            key=lambda x: x[1]['tempo_segundos'],
            reverse=True
        )
        
        for model_name, time_info in sorted_times:
            time_sec = time_info['tempo_segundos']
            time_min = time_info['tempo_minutos']
            
            # Formatar nome do modelo
            display_name = model_name.upper().replace('_', ' ')
            LOGGER.info(f"   {display_name:30s}: {time_sec:8.2f}s ({time_min:6.2f}min)")
        
        LOGGER.info(f"{'='*70}\n")
    
    def save_to_json(self):
        """Salva tempos em arquivo JSON."""
        output_file = os.path.join(self.metrics_dir, 'training_times.json')
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_times, f, indent=2, ensure_ascii=False)
            LOGGER.info(f"Tempos de treinamento salvos em: {output_file}")
            return output_file
        except Exception as e:
            LOGGER.error(f"Erro ao salvar tempos de treinamento: {e}")
            return None
    
    def get_summary_text(self) -> str:
        """Retorna um texto formatado com o resumo dos tempos."""
        if not self.training_times:
            return ""
        
        lines = [
            f"\n{'='*70}",
            "⏱️  RESUMO DE TEMPO DE TREINAMENTO - TODOS OS MODELOS",
            f"{'='*70}"
        ]
        
        sorted_times = sorted(
            self.training_times.items(),
            key=lambda x: x[1]['tempo_segundos'],
            reverse=True
        )
        
        for model_name, time_info in sorted_times:
            time_sec = time_info['tempo_segundos']
            time_min = time_info['tempo_minutos']
            display_name = model_name.upper().replace('_', ' ')
            lines.append(f"   {display_name:30s}: {time_sec:8.2f}s ({time_min:6.2f}min)")
        
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


# Instância global
_global_tracker = None


def get_tracker() -> TrainingTimeTracker:
    """Obtém ou cria a instância global do rastreador."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TrainingTimeTracker()
    return _global_tracker


def register_training_time(model_name: str, elapsed_time: float):
    """Função auxiliar para registrar tempo de treinamento."""
    tracker = get_tracker()
    tracker.register_model_time(model_name, elapsed_time)


def log_all_training_times():
    """Função auxiliar para exibir todos os tempos registrados."""
    tracker = get_tracker()
    tracker.log_training_summary()
    tracker.save_to_json()
