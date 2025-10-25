import argparse
import os
import glob
import pandas as pd
from config import RESULTS_DIR, LOGGER
import gerar_graficos as gg


def resolve_results_dir() -> str:
    if os.path.isabs(RESULTS_DIR):
        return RESULTS_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, RESULTS_DIR)


def find_history_file(history_dir: str, model_name: str) -> str | None:
    candidate = os.path.join(history_dir, f"{model_name}_history.csv")
    if os.path.exists(candidate):
        return candidate
    pattern_strict = os.path.join(history_dir, f"{model_name}_history*.csv")
    matches = sorted(glob.glob(pattern_strict))
    if matches:
        return matches[0]
    pattern_any = os.path.join(history_dir, "*history*.csv")
    matches = sorted(glob.glob(pattern_any))
    if matches:
        return matches[0]
    return None


def list_history_files(history_dir: str):
    try:
        entries = sorted(os.listdir(history_dir))
        LOGGER.info(f"History dir: {history_dir}")
        for e in entries:
            LOGGER.info(e)
    except Exception as e:
        LOGGER.error(f"Cannot list history dir: {e}")


def main(model_name: str, history_path_cli: str | None, list_only: bool):
    results_dir_abs = resolve_results_dir()
    history_dir = os.path.join(results_dir_abs, "history")
    if not os.path.isdir(history_dir):
        LOGGER.error(f"History directory not found: {history_dir}")
        raise SystemExit(1)
    if list_only:
        list_history_files(history_dir)
        return
    history_path = None
    if history_path_cli:
        history_path = history_path_cli if os.path.isabs(history_path_cli) else os.path.join(history_dir, history_path_cli)
    else:
        history_path = find_history_file(history_dir, model_name)
    if not history_path or not os.path.exists(history_path):
        LOGGER.error(f"History file not found for model '{model_name}'")
        list_history_files(history_dir)
        raise SystemExit(1)
    LOGGER.info(f"Using history file: {history_path}")
    df = pd.read_csv(history_path)
    gg.visualizar_historico_treinamento(df, model_name)
    LOGGER.info("History plots generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MLP")
    parser.add_argument("--history-path", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    main(args.model, args.history_path, args.list)
