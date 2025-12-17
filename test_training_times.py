#!/usr/bin/env python3
"""
Script de teste para demonstrar o novo sistema de rastreamento de tempos de treinamento.
"""

import sys
import os

# Adicionar o diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_time_tracker import get_tracker, log_all_training_times
from config import LOGGER

def test_tracking():
    """Testa o sistema de rastreamento de tempos."""
    
    LOGGER.info("\n" + "="*70)
    LOGGER.info("🧪 TESTE DO SISTEMA DE RASTREAMENTO DE TEMPOS DE TREINAMENTO")
    LOGGER.info("="*70 + "\n")
    
    tracker = get_tracker()
    
    # Simular tempos de treinamento
    test_times = {
        "MLP_Bayesian_Selected": 427.53,
        "CNN_Bayesian_Selected": 535.16,
        "Random Forest": 8.45,
        "Gradient Boosting": 12.32,
        "MLP": 123.45,
        "CNN": 456.78,
    }
    
    LOGGER.info("Registrando tempos de teste...\n")
    
    for model_name, elapsed_time in test_times.items():
        tracker.register_model_time(model_name, elapsed_time)
        LOGGER.info(f"  ✓ Registrado: {model_name:30s} = {elapsed_time:7.2f}s")
    
    LOGGER.info("\n")
    
    # Exibir resumo
    log_all_training_times()
    
    # Obter texto de resumo
    summary_text = tracker.get_summary_text()
    LOGGER.info("Texto de resumo (pode ser usado em relatórios):")
    LOGGER.info(summary_text)
    
    LOGGER.info("✅ Teste concluído com sucesso!")
    LOGGER.info(f"Arquivo JSON salvo em: resultados_diabetes/tempo_treinamento/training_times.json\n")

if __name__ == "__main__":
    try:
        test_tracking()
    except Exception as e:
        LOGGER.error(f"Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
