from fastapi import FastAPI, HTTPException
import pandas as pd

from .schemas import PredictRequest, PredictResponse, ModelInfo
from .model_loader import ModelHolder

app = FastAPI(title="Wine Quality API")


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    """Возвращает информацию о текущей модели из MLflow."""
    info = ModelHolder.get_model_info()
    return ModelInfo(**info)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = ModelHolder.get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    df = pd.DataFrame(
        [s.model_dump(by_alias=True) for s in request.samples]
    )

    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    preds = [float(p) for p in preds]

    return PredictResponse(predictions=preds)
