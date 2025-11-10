from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class WineFeatures(BaseModel):

    model_config = ConfigDict(
        validate_by_name=True
    )

    fixed_acidity: float = Field(..., alias="fixed acidity")
    volatile_acidity: float = Field(..., alias="volatile acidity")
    citric_acid: float = Field(..., alias="citric acid")
    residual_sugar: float = Field(..., alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float


class PredictRequest(BaseModel):
    samples: List[WineFeatures]


class PredictResponse(BaseModel):
    predictions: List[float]


class ModelInfo(BaseModel):
    model_name: str
    version: str
    trained_at: Optional[str] = None
    metrics: Optional[dict] = None
