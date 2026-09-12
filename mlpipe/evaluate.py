import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from mlpipe.config import get_project_root, load_config
from mlpipe.pipeline import get_all_models


def evaluate(model_name: str = "lightgbm"):
    root = get_project_root()
    config = load_config()

    model_path = root / "models" / f"{model_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run mlpipe.train first."
        )
    print(f"Loading trained model from: {model_path}")
    model = joblib.load(model_path)
    
    processed_data_path = Path(config["data"]["processed_data"])
    df = pd.read_csv(processed_data_path)
    
    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Evaluating {model_name} on {X_test.shape[0]} test samples...")
    
    y_test_pred = model.predict(X_test)
    
    y_test_dollars = np.exp(y_test)
    y_pred_dollars = np.exp(y_test_pred)
    
    test_r2_log = round(r2_score(y_test, y_test_pred), 4)
    test_mae = round(mean_absolute_error(y_test_dollars, y_pred_dollars), 2)
    test_rmse = round(np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars)), 2)
    test_mape = round(mean_absolute_percentage_error(y_test_dollars, y_pred_dollars) * 100, 2)
    
    metrics = {
        "model": model_name,
        "test_r2_log": test_r2_log,
        "test_mae_dollars": test_mae,
        "test_rmse_dollars": test_rmse,
        "test_mape_percent": test_mape,
    }
    
    print(f"{model_name.upper()} - R2: {test_r2_log} | MAE: ${test_mae:,.2f} | RMSE: ${test_rmse:,.2f} | MAPE: {test_mape}%")

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = reports_dir / "metrics.json"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = y_test_dollars - y_pred_dollars
    ax.scatter(y_pred_dollars / 1000, residuals / 1000, alpha=0.3, color="royalblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Plot - {model_name.upper()}")
    ax.set_xlabel("Predicted Price ($ in Thousands)")
    ax.set_ylabel("Residual Error ($ in Thousands)")
    plt.tight_layout()

    plot_path = reports_dir / f"residual_plot_{model_name}.png"
    fig.savefig(plot_path)
    plt.close(fig)

    mlflow.set_experiment("KingCounty_House_Valuation")
    run_name = f"EVAL_{model_name.upper()}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "test_r2_log": test_r2_log,
            "test_mae_dollars": test_mae,
            "test_rmse_dollars": test_rmse,
            "test_mape_percent": test_mape,
        })
        mlflow.log_artifact(str(metrics_file))
        mlflow.log_artifact(str(plot_path))

    return metrics


def evaluate_all():
    root = get_project_root()
    config = load_config()
    models = get_all_models(config)
    results = []

    for name in models.keys():
        model_file = root / "models" / f"{name}_model.joblib"
        if not model_file.exists():
            continue
        try:
            m = evaluate(model_name=name)
            results.append(m)
        except Exception as e:
            print(f"Failed {name}: {e}")

    if results:
        df_results = pd.DataFrame(results).sort_values(by="test_rmse_dollars").reset_index(drop=True)
        summary_path = root / "reports" / "benchmark_leaderboard.csv"
        df_results.to_csv(summary_path, index=False)
        print(df_results.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        evaluate_all()
    else:
        evaluate(model_name=args.model)
