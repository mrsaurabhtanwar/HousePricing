import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_all_models(config: dict) -> dict:
    lin_cfg = config["linear_models"]
    random_state = config["data"]["random_state"]

    cv_strategy = KFold(
        n_splits=lin_cfg["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )
    alphas_grid = np.logspace(
        lin_cfg["alpha_grid"]["start_exp"],
        lin_cfg["alpha_grid"]["stop_exp"],
        lin_cfg["alpha_grid"]["num_points"],
    )

    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=lin_cfg["alpha"], random_state=random_state))
        ]),
        "ridge_cv": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=alphas_grid, cv=cv_strategy))
        ]),
        "lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=lin_cfg["alpha"],
                max_iter=lin_cfg["max_iters"],
                random_state=random_state
            ))
        ]),
        "lasso_cv": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(
                alphas=alphas_grid,
                cv=cv_strategy,
                max_iter=lin_cfg["max_iters"],
                random_state=random_state
            ))
        ]),
        "elastic_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=lin_cfg["alpha"],
                l1_ratio=lin_cfg["elastic_net"]["l1_ratio"],
                max_iter=lin_cfg["max_iters"],
                random_state=random_state
            ))
        ]),
        "elastic_net_cv": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(
                alphas=alphas_grid,
                cv=cv_strategy,
                l1_ratio=lin_cfg["elastic_net"]["l1_ratios"],
                max_iter=lin_cfg["max_iters"],
                random_state=random_state
            ))
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(**config["distance_models"]["knn"]))
        ]),
        "linear_svr": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVR(
                dual="auto",
                C=config["distance_models"]["linear_svr"]["c_param"],
                tol=config["distance_models"]["linear_svr"]["tol"],
                max_iter=config["distance_models"]["linear_svr"]["max_iters"],
                random_state=random_state
            ))
        ]),

        "decision_tree": DecisionTreeRegressor(**config["tree_models"]["decision_tree"]),
        "random_forest": RandomForestRegressor(**config["tree_models"]["random_forest"]),
        "extra_trees": ExtraTreesRegressor(**config["tree_models"]["extra_trees"]),

        "gradient_boosting": GradientBoostingRegressor(**config["boosting_models"]["gradient_boosting"]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(**config["boosting_models"]["hist_gradient_boosting"]),
        "xgboost": XGBRegressor(**config["boosting_models"]["xgboost"]),
        "lightgbm": LGBMRegressor(**config["boosting_models"]["lightgbm"]),
        "catboost": CatBoostRegressor(**config["boosting_models"]["catboost"]),
    }

    return models


def get_model(model_name: str, config: dict):
    models = get_all_models(config)
    if model_name not in models:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Available: {list(models.keys())}"
        )

    return models[model_name]