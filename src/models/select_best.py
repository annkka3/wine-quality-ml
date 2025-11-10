import pathlib
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple

import mlflow
import mlflow.sklearn

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model_best.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
METRICS_HISTORY_PATH = ROOT / "models" / "metrics_history.json"

# Пороги для определения деградации (в процентах)
DEGRADATION_THRESHOLD_RMSE = 5.0  # 5% ухудшение RMSE считается деградацией


def extract_metrics(run_data) -> Dict[str, float]:
    """Извлекает метрики из run данных."""
    metrics = {}
    for col in run_data.index:
        if col.startswith("metrics."):
            metric_name = col.replace("metrics.", "")
            value = run_data[col]
            if value is not None and not (isinstance(value, float) and value != value):  # NaN check
                metrics[metric_name] = float(value)
    return metrics


def extract_params(run_data) -> Dict[str, Any]:
    """Извлекает параметры из run данных."""
    params = {}
    for col in run_data.index:
        if col.startswith("params."):
            param_name = col.replace("params.", "")
            value = run_data[col]
            if value is not None:
                # Пытаемся преобразовать в число, если возможно
                try:
                    if '.' in str(value):
                        params[param_name] = float(value)
                    else:
                        params[param_name] = int(value)
                except (ValueError, TypeError):
                    params[param_name] = str(value)
    return params


def get_model_type_from_name(run_name: str) -> str:
    """Определяет тип модели по имени run."""
    run_name_lower = run_name.lower()
    if "random_forest" in run_name_lower or "rf" in run_name_lower:
        return "RandomForestRegressor"
    elif "gbrt" in run_name_lower or "gradient" in run_name_lower or "gb" in run_name_lower:
        return "GradientBoostingRegressor"
    else:
        return "Unknown"


def load_previous_metadata() -> Dict[str, Any]:
    """Загружает метаданные предыдущей лучшей модели."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def load_metrics_history() -> list:
    """Загружает историю метрик."""
    if METRICS_HISTORY_PATH.exists():
        try:
            with open(METRICS_HISTORY_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_metrics_history(history: list):
    """Сохраняет историю метрик."""
    with open(METRICS_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def check_degradation(current_rmse: float, previous_rmse: float) -> Tuple[bool, float]:
    """
    Проверяет деградацию метрики RMSE.
    
    Returns:
        (is_degraded, degradation_percent): True если деградация превышает порог
    """
    if previous_rmse is None:
        return False, 0.0
    
    degradation_percent = ((current_rmse - previous_rmse) / previous_rmse) * 100
    is_degraded = degradation_percent > DEGRADATION_THRESHOLD_RMSE
    
    return is_degraded, degradation_percent


def main():
    """
    Выбирает лучшую модель из последних экспериментов MLflow
    на основе метрики RMSE (меньше = лучше).
    Сохраняет модель и метаданные (метрики, параметры, дата обучения).
    """
    mlflow.set_experiment("wine_quality")
    
    # Получаем все последние запуски эксперимента
    experiment = mlflow.get_experiment_by_name("wine_quality")
    if experiment is None:
        raise RuntimeError("Experiment 'wine_quality' not found. Run training first.")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=10
    )
    
    if runs.empty:
        raise RuntimeError("No runs found in experiment. Run training first.")
    
    # Берем лучший run (с наименьшим RMSE)
    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_rmse = best_run.get("metrics.rmse", None)
    best_model_name = best_run.get("tags.mlflow.runName", "unknown")
    
    # Проверяем наличие метрик
    if best_rmse is None or (isinstance(best_rmse, float) and best_rmse != best_rmse):  # NaN check
        raise RuntimeError(f"Best run {best_run_id} has invalid RMSE metric.")
    
    print(f"Best model: {best_model_name}")
    print(f"Run ID: {best_run_id}")
    print(f"RMSE: {best_rmse:.4f}")
    
    # Загружаем предыдущие метаданные для сравнения
    previous_metadata = load_previous_metadata()
    previous_rmse = None
    if previous_metadata and "metrics" in previous_metadata:
        previous_rmse = previous_metadata["metrics"].get("rmse")
    
    # Проверяем деградацию
    is_degraded, degradation_percent = check_degradation(best_rmse, previous_rmse)
    
    if previous_rmse is not None:
        print(f"\n=== Model Comparison ===")
        print(f"Previous RMSE: {previous_rmse:.4f}")
        print(f"Current RMSE:  {best_rmse:.4f}")
        if best_rmse < previous_rmse:
            improvement = ((previous_rmse - best_rmse) / previous_rmse) * 100
            print(f"✅ Improvement: {improvement:.2f}%")
        else:
            print(f"⚠️  Degradation: {degradation_percent:.2f}%")
            if is_degraded:
                print(f"🚨 WARNING: Model degradation exceeds threshold ({DEGRADATION_THRESHOLD_RMSE}%)!")
                print(f"   Consider reviewing the training process or data quality.")
    
    # Извлекаем все метрики и параметры
    metrics = extract_metrics(best_run)
    params = extract_params(best_run)
    
    # Получаем время обучения
    start_time = best_run.get("start_time", None)
    trained_at = None
    if start_time is not None:
        try:
            if not (isinstance(start_time, float) and start_time != start_time):  # NaN check
                trained_at = datetime.fromtimestamp(start_time / 1000).isoformat()
        except (ValueError, TypeError, OverflowError):
            pass
    
    # Определяем тип модели
    model_type = get_model_type_from_name(best_model_name)
    
    # Загружаем модель из MLflow
    model_uri = f"runs:/{best_run_id}/model"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow: {e}")
    
    # Сохраняем модель
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")
    
    # Создаем метаданные
    metadata = {
        "model_name": model_type,
        "run_name": best_model_name,
        "run_id": best_run_id,
        "version": "1.0.0",
        "trained_at": trained_at,
        "selected_at": datetime.now().isoformat(),
        "metrics": metrics,
        "parameters": params,
        "selection_criterion": "lowest_rmse",
        "degradation_detected": is_degraded,
        "degradation_percent": degradation_percent if previous_rmse else None,
        "previous_rmse": previous_rmse,
    }
    
    # Сохраняем метаданные
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {METADATA_PATH}")
    
    # Обновляем историю метрик
    history = load_metrics_history()
    history_entry = {
        "selected_at": metadata["selected_at"],
        "run_id": best_run_id,
        "model_name": model_type,
        "rmse": best_rmse,
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "degradation_percent": degradation_percent if previous_rmse else None,
    }
    history.append(history_entry)
    # Оставляем только последние 50 записей
    history = history[-50:]
    save_metrics_history(history)
    print(f"Updated metrics history ({len(history)} entries)")
    
    # Выводим краткую сводку
    print("\n=== Model Summary ===")
    print(f"Model Type: {model_type}")
    print(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
    print(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
    print(f"MSE: {metrics.get('mse', 'N/A'):.4f}")
    if trained_at:
        print(f"Trained at: {trained_at}")
    print(f"Selected at: {metadata['selected_at']}")
    
    if is_degraded:
        print(f"\n⚠️  ALERT: Model degradation detected!")
        return 1  # Возвращаем код ошибки для CI/CD
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)

