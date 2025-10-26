import os
from bayesian_tuning import (
    analyze_trial_selection,
    compare_all_trials,
    analyze_hyperparameter_correlations,
    export_comparison_report
)
from config import RESULTS_DIR, LOGGER

def main():
    results_dir = os.path.join(RESULTS_DIR, 'tuning', 'bayesian_results')

    if not os.path.exists(results_dir):
        LOGGER.error(f"Diretório não encontrado: {results_dir}")
        LOGGER.info("Execute primeiro a otimização bayesiana antes de analisar os resultados.")
        return

    LOGGER.info("=" * 80)
    LOGGER.info("INICIANDO ANÁLISE COMPLETA DOS RESULTADOS BAYESIANOS")
    LOGGER.info("=" * 80)

    analysis_results = analyze_trial_selection(results_dir)

    if analysis_results:
        LOGGER.info(f"\n✓ Análise de seleção concluída")
        LOGGER.info(f"  Melhor trial: #{analysis_results['best_trial']}")
        LOGGER.info(f"  Percentil: {analysis_results['percentile']:.1f}%")

    df = compare_all_trials(results_dir, metric='best_val_precision')
    LOGGER.info(f"\n✓ Comparação entre trials concluída ({len(df)} trials analisados)")

    correlations = analyze_hyperparameter_correlations(results_dir)
    LOGGER.info(f"\n✓ Análise de correlações concluída")

    report_path = export_comparison_report(results_dir)
    LOGGER.info(f"\n✓ Relatório exportado para: {report_path}")

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("ANÁLISE COMPLETA FINALIZADA")
    LOGGER.info("=" * 80)

if __name__ == "__main__":
    main()

