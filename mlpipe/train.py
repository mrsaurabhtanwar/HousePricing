from pathlib import Path
import time
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from mlpipe.config import get_project_root, load_config
from mlpipe.pipeline import get_all_models, get_model


def extract_model_params(model_name: str, config: dict) -> dict:
    """Return MLflow-friendly parameters for one canonical model name."""
    data_cfg = config["data"]
    linear_cfg = config["linear_models"]
    distance_cfg = config["distance_models"]

    params = {
        "model_name": model_name,
        "test_size": data_cfg["test_size"],
        "random_state": data_cfg["random_state"],
        "target_column": data_cfg["target_column"],
    }

    tree_models = {
        "decision_tree", 
        "random_forest", 
        "extra_trees"
    }
    
    boosting_models = {
        "gradient_boosting",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    
    valid_names = {
        "linear_regression",
        "ridge",
        "ridge_cv",
        "lasso",
        "lasso_cv",
        "elastic_net",
        "elastic_net_cv",
        "knn",
        "linear_svr",
        *tree_models,
        *boosting_models,
    }

    if model_name == "linear_regression":
        params["scaler"] = "StandardScaler"
    elif model_name == "ridge":
        params.update(alpha=linear_cfg["alpha"], scaler="StandardScaler")
    elif model_name == "ridge_cv":
        params.update(
            cv_folds=linear_cfg["cv_folds"],
            alpha_grid_start=linear_cfg["alpha_grid"]["start_exp"],
            alpha_grid_stop=linear_cfg["alpha_grid"]["stop_exp"],
            alpha_grid_points=linear_cfg["alpha_grid"]["num_points"],
            scaler="StandardScaler",
        )
    elif model_name == "lasso":
        params.update(
            alpha=linear_cfg["alpha"],
            max_iter=linear_cfg["max_iters"],
            scaler="StandardScaler",
        )
    elif model_name == "lasso_cv":
        params.update(
            cv_folds=linear_cfg["cv_folds"],
            max_iter=linear_cfg["max_iters"],
            alpha_grid_points=linear_cfg["alpha_grid"]["num_points"],
            scaler="StandardScaler",
        )
    elif model_name == "elastic_net":
        params.update(
            alpha=linear_cfg["alpha"],
            l1_ratio=linear_cfg["elastic_net"]["l1_ratio"],
            max_iter=linear_cfg["max_iters"],
            scaler="StandardScaler",
        )
    elif model_name == "elastic_net_cv":
        params.update(
            cv_folds=linear_cfg["cv_folds"],
            max_iter=linear_cfg["max_iters"],
            l1_ratios=str(linear_cfg["elastic_net"]["l1_ratios"]),
            alpha_grid_points=linear_cfg["alpha_grid"]["num_points"],
            scaler="StandardScaler",
        )
    elif model_name == "knn":
        params.update(distance_cfg["knn"], scaler="StandardScaler")
    elif model_name == "linear_svr":
        svr_cfg = distance_cfg["linear_svr"]
        params.update(
            C=svr_cfg["c_param"],
            tol=svr_cfg["tol"],
            max_iter=svr_cfg["max_iters"],
            dual="auto",
            scaler="StandardScaler",
        )
    elif model_name in tree_models:
        params.update(config["tree_models"][model_name])
    elif model_name in boosting_models:
        params.update(config["boosting_models"][model_name])
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Available: {sorted(valid_names)}"
        )

    return params



def train(model_name: str):
    root = get_project_root()
    config = load_config()
    
    processed_data_path = Path(config["data"]["processed_data"])
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found at: {processed_data_path}. Run mlpipe.prepare first."           
        )
    
    print(f"Loading processed dataset: {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    target_col = config["data"]["target_column"]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    test_size = config["data"]["test_size"]
    rand_state = config["data"]["random_state"]
    
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=rand_state
    )
    print(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    
    mlflow.set_experiment("KingCounty_House_Valuation")
    run_name = f"TRAIN_{model_name.upper()}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Active MLflow Run ID: {run.info.run_id}")
        
        model = get_model(model_name, config)
        
        model_params = extract_model_params(model_name, config)
        mlflow.log_params(model_params)
        
        print(f"Training {model_name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        fit_time = round(time.time() - start_time)
        print(f"Training completed in {fit_time}s")
        mlflow.log_metric("train_time_sec", fit_time)
        
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner_estimator = model.named_steps["model"]
            if hasattr(inner_estimator, "alpha_"):
                mlflow.log_metric("best_alpha", float(inner_estimator.alpha_))
            if hasattr(inner_estimator, "l1_ratio_"):
                mlflow.log_metric("best_l1_ratio", float(inner_estimator.l1_ratio_))
                
                
        sample_input = X_train.head(5)
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)
        
        models_dir = root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = models_dir / f"{model_name}_model.joblib"
        joblib.dump(model, artifact_path)
        print(f"Model successfully saved to: {artifact_path}")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=sample_input.head(1),
            registered_model_name=f"HousePricing_{model_name.upper()}",
        )
        print(f"Model successfully registered in MLflow: HousePricing_{model_name.upper()}")


def train_all():
    config = load_config()
    models = get_all_models(config)
    for name in models.keys():
        try:
            train(model_name=name)
        except Exception as e:
            print(f"Failed {name}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        train_all()
    else:
        train(model_name=args.model)
