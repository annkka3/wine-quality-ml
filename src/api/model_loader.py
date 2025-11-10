import pathlib
import json
import joblib
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import mlflow
    import pandas as pd
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    pd = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model_best.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


class ModelHolder:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            if not MODEL_PATH.exists():
                raise RuntimeError(
                    f"Model file not found at {MODEL_PATH}. "
                    f"Run training and/or `dvc pull` first."
                )
            cls._model = joblib.load(MODEL_PATH)
        return cls._model

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """
        Получает информацию о модели.
        Приоритет: 1) JSON метаданные, 2) MLflow, 3) базовая информация.
        """
        # Сначала пытаемся загрузить из JSON метаданных
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r") as f:
                    metadata = json.load(f)
                
                # Преобразуем в формат, ожидаемый API
                return {
                    "model_name": metadata.get("model_name", "Unknown"),
                    "version": metadata.get("version", "1.0.0"),
                    "trained_at": metadata.get("trained_at"),
                    "metrics": metadata.get("metrics"),
                }
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Could not load model metadata from JSON: {e}")
        
        # Fallback на MLflow
        if MLFLOW_AVAILABLE:
            try:
                return cls._get_info_from_mlflow()
            except Exception as e:
                print(f"Warning: Could not load model info from MLflow: {e}")
        
        # Последний fallback - базовая информация
        return cls._get_basic_info()
    
    @classmethod
    def _get_info_from_mlflow(cls) -> Dict[str, Any]:
        """Получает информацию о модели из MLflow."""
        mlflow.set_experiment("wine_quality")
        experiment = mlflow.get_experiment_by_name("wine_quality")
        
        if experiment is None:
            raise RuntimeError("Experiment not found")
        
        # Ищем лучший run (с наименьшим RMSE)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.rmse ASC"],
            max_results=1
        )
        
        if runs.empty:
            raise RuntimeError("No runs found")
        
        best_run = runs.iloc[0]
        run_name = best_run.get("tags.mlflow.runName", "unknown")
        
        # Получаем метрики
        metrics = {
            "rmse": float(best_run.get("metrics.rmse", 0.0)),
            "mae": float(best_run.get("metrics.mae", 0.0)),
            "mse": float(best_run.get("metrics.mse", 0.0)),
        }
        
        # Определяем тип модели по имени
        model_name = "RandomForestRegressor" if "random_forest" in run_name.lower() else "GradientBoostingRegressor"
        if "gbrt" in run_name.lower() or "gradient" in run_name.lower():
            model_name = "GradientBoostingRegressor"
        
        # Получаем время обучения
        start_time = best_run.get("start_time", None)
        trained_at = None
        if start_time is not None:
            try:
                # Проверяем, является ли значение NaN (pandas или numpy)
                if pd is not None and hasattr(pd, 'isna') and pd.isna(start_time):
                    pass
                elif start_time != start_time:  # NaN check для обычных float
                    pass
                else:
                    trained_at = datetime.fromtimestamp(start_time / 1000).isoformat()
            except (ValueError, TypeError, OverflowError):
                pass
        
        return {
            "model_name": model_name,
            "version": "1.0.0",
            "trained_at": trained_at,
            "metrics": metrics,
        }
    
    @classmethod
    def _get_basic_info(cls) -> Dict[str, Any]:
        """Возвращает базовую информацию о модели."""
        model = cls.get_model()
        model_type = type(model).__name__
        
        return {
            "model_name": model_type,
            "version": "1.0.0",
            "trained_at": None,
            "metrics": None,
        }
