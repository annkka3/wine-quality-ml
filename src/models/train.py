import argparse
import pathlib
import yaml
import joblib
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

import mlflow
import mlflow.sklearn

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "winequality-red.csv"
PARAMS_PATH = ROOT / "params.yaml"


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def get_model(name: str, params: dict):
    """Создает модель с заданными параметрами."""
    if name == "random_forest":
        return RandomForestRegressor(**params)
    elif name == "gbrt":
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_param_grid(name: str, params: dict) -> dict:
    """Возвращает сетку параметров для GridSearchCV."""
    base_params = {k: v for k, v in params.items() if k != "random_state"}
    
    if name == "random_forest":
        return {
            "n_estimators": params.get("grid_search", {}).get("n_estimators", [100, 150, 200]),
            "max_depth": params.get("grid_search", {}).get("max_depth", [6, 8, 10, None]),
            "min_samples_split": params.get("grid_search", {}).get("min_samples_split", [2, 5, 10]),
        }
    elif name == "gbrt":
        return {
            "n_estimators": params.get("grid_search", {}).get("n_estimators", [200, 300, 400]),
            "learning_rate": params.get("grid_search", {}).get("learning_rate", [0.01, 0.05, 0.1]),
            "max_depth": params.get("grid_search", {}).get("max_depth", [3, 5, 7]),
        }
    else:
        return {}


def main(model_name: str, use_grid_search: bool = True):
    """
    Обучает модель с кросс-валидацией и опциональным GridSearchCV.
    
    Args:
        model_name: Имя модели (random_forest или gbrt)
        use_grid_search: Использовать ли GridSearchCV для подбора гиперпараметров
    """
    print("DATA_PATH:", DATA_PATH)
    params = load_params()

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]
    cv_folds = params["train"].get("cv_folds", 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model_params = params["models"][model_name]
    random_state_model = model_params.get("random_state", random_state)

    mlflow.set_experiment("wine_quality")

    run_name = f"{model_name}_gridsearch" if use_grid_search else model_name
    
    with mlflow.start_run(run_name=run_name):
        # Логируем базовые параметры
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("use_grid_search", use_grid_search)
        mlflow.log_param("cv_folds", cv_folds)
        
        if use_grid_search and "grid_search" in model_params:
            # Используем GridSearchCV
            print(f"[{model_name}] Starting GridSearchCV with {cv_folds}-fold CV...")
            
            base_model = get_model(model_name, {"random_state": random_state_model})
            param_grid = get_param_grid(model_name, model_params)
            
            # Создаем scorer для RMSE (меньше = лучше)
            rmse_scorer = make_scorer(
                lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                greater_is_better=False
            )
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=rmse_scorer,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            # Логируем лучшие параметры
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("best_cv_score", -grid_search.best_score_)  # Инвертируем, т.к. scorer отрицательный
            
            print(f"[{model_name}] Best parameters: {grid_search.best_params_}")
            print(f"[{model_name}] Best CV RMSE: {-grid_search.best_score_:.4f}")
            
            # Кросс-валидация на полном train set для дополнительных метрик
            cv_scores_rmse = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, 
                scoring=rmse_scorer,
                n_jobs=-1
            )
            cv_rmse_mean = -cv_scores_rmse.mean()
            cv_rmse_std = cv_scores_rmse.std()
            
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)
            
            print(f"[{model_name}] CV RMSE: {cv_rmse_mean:.4f} (+/- {cv_rmse_std:.4f})")
        else:
            # Обычное обучение без GridSearch
            print(f"[{model_name}] Training without GridSearch...")
            model = get_model(model_name, model_params)
            mlflow.log_params(model_params)
            
            # Кросс-валидация для оценки
            rmse_scorer = make_scorer(
                lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                greater_is_better=False
            )
            cv_scores_rmse = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring=rmse_scorer,
                n_jobs=-1
            )
            cv_rmse_mean = -cv_scores_rmse.mean()
            cv_rmse_std = cv_scores_rmse.std()
            
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)
            
            print(f"[{model_name}] CV RMSE: {cv_rmse_mean:.4f} (+/- {cv_rmse_std:.4f})")
        
        # Финальное обучение на всем train set
        model.fit(X_train, y_train)
        
        # Оценка на test set
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("mse", mse)  # Для обратной совместимости
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        print(f"[{model_name}] Test MSE:  {mse:.3f}")
        print(f"[{model_name}] Test RMSE: {rmse:.3f}")
        print(f"[{model_name}] Test MAE:  {mae:.3f}")

        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        out_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, out_path)
        print(f"Saved local model to {out_path}")

        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["random_forest", "gbrt"])
    parser.add_argument("--no-grid-search", action="store_true",
                        help="Disable GridSearchCV and use fixed parameters")
    args = parser.parse_args()
    main(args.model, use_grid_search=not args.no_grid_search)
