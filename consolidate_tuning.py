import os
import json
import re
import pandas as pd
from pathlib import Path


def _read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _is_valid_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    try:
        return not df.dropna(how="all").empty
    except Exception:
        return True


def _extract_metrics_from_kt_trial(trial_json):
    metrics = trial_json.get("metrics", {}).get("metrics", {})
    out = {}
    def _get(m):
        v = metrics.get(m, {}).get("observations", [])
        if not v:
            return None
        val = v[0].get("value")
        if isinstance(val, list) and val:
            return val[0]
        return val
    out["best_val_precision"] = _get("val_precision")
    out["best_val_accuracy"] = _get("val_accuracy")
    out["best_val_auc"] = _get("val_auc")
    out["best_val_recall"] = _get("val_recall")
    out["best_val_loss"] = _get("val_loss")
    out["best_val_pr_auc"] = _get("val_pr_auc")
    return out


def _flatten_hp(trial_json):
    values = trial_json.get("hyperparameters", {}).get("values", {})
    return dict(values)


def _parse_trial_number(name):
    m = re.search(r"(\d+)$", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def collect_from_keras_tuner_dir(dir_path, tuning_source):
    trials_rows = []
    epochs_rows = []
    trials_dir = Path(dir_path)
    for child in trials_dir.iterdir():
        if child.is_dir() and child.name.startswith("trial_"):
            trial_json_path = child / "trial.json"
            if not trial_json_path.exists():
                continue
            try:
                with open(trial_json_path, "r") as f:
                    trial = json.load(f)
            except Exception:
                continue
            row = {
                "tuning_source": tuning_source,
                "trial_id": trial.get("trial_id"),
                "trial_number": _parse_trial_number(child.name),
                "status": trial.get("status"),
                "duration_seconds": None,
            }
            row.update(_extract_metrics_from_kt_trial(trial))
            row.update(_flatten_hp(trial))
            trials_rows.append(row)
            metrics = trial.get("metrics", {}).get("metrics", {})
            best_step = trial.get("best_step")
            epoch_row = {
                "tuning_source": tuning_source,
                "trial_id": trial.get("trial_id"),
                "trial_number": _parse_trial_number(child.name),
                "epoch": best_step,
            }
            for k in [
                "val_precision",
                "val_pr_auc",
                "val_accuracy",
                "val_auc",
                "val_recall",
                "val_loss",
            ]:
                obs = metrics.get(k, {}).get("observations", [])
                if obs:
                    v = obs[0].get("value")
                    if isinstance(v, list) and v:
                        v = v[0]
                    epoch_row[k] = v
                else:
                    epoch_row[k] = None
            epochs_rows.append(epoch_row)
    df_trials = pd.DataFrame(trials_rows) if trials_rows else pd.DataFrame()
    df_epochs = pd.DataFrame(epochs_rows) if epochs_rows else pd.DataFrame()
    return df_trials, df_epochs


def _filter_valid_dfs(dfs):
    out = []
    for df in dfs:
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        if df.empty:
            continue
        try:
            df2 = df.dropna(how="all")
        except Exception:
            df2 = df
        if df2.empty:
            continue
        out.append(df2)
    return out


def _infer_model_type_from_path(path_str: str) -> str:
    s = str(path_str).lower()
    tokens_cnn = ["cnn", "conv"]
    tokens_mlp = ["mlp", "dense", "bayesian_mlp", "bayesian_results"]
    for t in tokens_cnn:
        if t in s:
            return "cnn"
    for t in tokens_mlp:
        if t in s and "cnn" not in s:
            return "mlp"
    return "unknown"


def _ensure_meta_columns(df: pd.DataFrame, tuning_source: str) -> pd.DataFrame:
    if "tuning_source" not in df.columns:
        df.insert(0, "tuning_source", tuning_source)
    if "model_type" not in df.columns:
        mt = _infer_model_type_from_path(tuning_source)
        insert_pos = 1 if "tuning_source" in df.columns else 0
        df.insert(insert_pos, "model_type", mt)
    return df


def consolidate_tuning(root_dir):
    root = Path(root_dir)
    out_dir = root
    all_trials = []
    all_epochs = []
    for base, dirs, files in os.walk(root_dir):
        base_path = Path(base)
        if "oracle.json" in files and any(d.startswith("trial_") for d in dirs):
            tuning_source = str(base_path.relative_to(root))
            df_t, df_e = collect_from_keras_tuner_dir(base_path, tuning_source)
            if _is_valid_df(df_t):
                df_t = _ensure_meta_columns(df_t, tuning_source)
                all_trials.append(df_t)
            if _is_valid_df(df_e):
                df_e = _ensure_meta_columns(df_e, tuning_source)
                all_epochs.append(df_e)
        if "bayesian_trials_summary.csv" in files:
            csv_path = base_path / "bayesian_trials_summary.csv"
            df = _read_csv_safe(csv_path)
            if _is_valid_df(df):
                tuning_source = str(base_path.relative_to(root))
                df = _ensure_meta_columns(df, tuning_source)
                all_trials.append(df)
            histories_dir = base_path / "trial_histories"
            if histories_dir.exists() and histories_dir.is_dir():
                for hfile in histories_dir.glob("*.csv"):
                    dfh = _read_csv_safe(hfile)
                    if _is_valid_df(dfh):
                        tuning_source = str(base_path.relative_to(root))
                        dfh = _ensure_meta_columns(dfh, tuning_source)
                        all_epochs.append(dfh)
        if "trials_summary.csv" in files:
            csv_path = base_path / "trials_summary.csv"
            df = _read_csv_safe(csv_path)
            if _is_valid_df(df):
                tuning_source = str(base_path.relative_to(root))
                df = _ensure_meta_columns(df, tuning_source)
                all_trials.append(df)
    all_trials = _filter_valid_dfs(all_trials)
    all_epochs = _filter_valid_dfs(all_epochs)
    if not all_trials:
        df_all_trials = pd.DataFrame()
    elif len(all_trials) == 1:
        df_all_trials = all_trials[0].copy()
    else:
        df_all_trials = pd.concat(all_trials, ignore_index=True)
    if not all_epochs:
        df_all_epochs = pd.DataFrame()
    elif len(all_epochs) == 1:
        df_all_epochs = all_epochs[0].copy()
    else:
        df_all_epochs = pd.concat(all_epochs, ignore_index=True)
    out_trials = out_dir / "consolidated_trials_summary.csv"
    out_epochs = out_dir / "consolidated_epoch_history.csv"
    if not df_all_trials.empty:
        df_all_trials.to_csv(out_trials, index=False)
    else:
        pd.DataFrame().to_csv(out_trials, index=False)
    if not df_all_epochs.empty:
        df_all_epochs.to_csv(out_epochs, index=False)
    else:
        pd.DataFrame().to_csv(out_epochs, index=False)
    out_trials_mlp = out_dir / "consolidated_trials_summary_mlp.csv"
    out_trials_cnn = out_dir / "consolidated_trials_summary_cnn.csv"
    out_epochs_mlp = out_dir / "consolidated_epoch_history_mlp.csv"
    out_epochs_cnn = out_dir / "consolidated_epoch_history_cnn.csv"
    if not df_all_trials.empty and "model_type" in df_all_trials.columns:
        df_mlp_t = df_all_trials[df_all_trials["model_type"] == "mlp"]
        df_cnn_t = df_all_trials[df_all_trials["model_type"] == "cnn"]
    else:
        df_mlp_t = pd.DataFrame()
        df_cnn_t = pd.DataFrame()
    if not df_all_epochs.empty and "model_type" in df_all_epochs.columns:
        df_mlp_e = df_all_epochs[df_all_epochs["model_type"] == "mlp"]
        df_cnn_e = df_all_epochs[df_all_epochs["model_type"] == "cnn"]
    else:
        df_mlp_e = pd.DataFrame()
        df_cnn_e = pd.DataFrame()
    df_mlp_t.to_csv(out_trials_mlp, index=False)
    df_cnn_t.to_csv(out_trials_cnn, index=False)
    df_mlp_e.to_csv(out_epochs_mlp, index=False)
    df_cnn_e.to_csv(out_epochs_cnn, index=False)
    return str(out_trials), str(out_epochs)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "resultados_diabetes", "tuning")
    t, e = consolidate_tuning(base)
    print(t)
    print(e)
    print(os.path.join(base, "consolidated_trials_summary_mlp.csv"))
    print(os.path.join(base, "consolidated_trials_summary_cnn.csv"))
    print(os.path.join(base, "consolidated_epoch_history_mlp.csv"))
    print(os.path.join(base, "consolidated_epoch_history_cnn.csv"))
