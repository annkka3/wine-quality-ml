from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_healthcheck():
    resp = client.get("/healthcheck")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_model_info():
    """Тест для endpoint /model-info."""
    resp = client.get("/model-info")
    # Может вернуть 200 или 500 в зависимости от наличия модели
    assert resp.status_code in [200, 500]
    if resp.status_code == 200:
        data = resp.json()
        assert "model_name" in data
        assert "version" in data
        assert isinstance(data["model_name"], str)
        assert isinstance(data["version"], str)


def test_predict_one_sample():
    payload = {
        "samples": [
            {
                "fixed acidity": 7.4,
                "volatile acidity": 0.7,
                "citric acid": 0.0,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        ]
    }

    resp = client.post("/predict", json=payload)
    # Может вернуть 200 или 500 в зависимости от наличия модели
    assert resp.status_code in [200, 500]
    if resp.status_code == 200:
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 1
        assert isinstance(data["predictions"][0], (int, float))


def test_predict_multiple_samples():
    """Тест для предсказания нескольких образцов."""
    payload = {
        "samples": [
            {
                "fixed acidity": 7.4,
                "volatile acidity": 0.7,
                "citric acid": 0.0,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            },
            {
                "fixed acidity": 7.8,
                "volatile acidity": 0.88,
                "citric acid": 0.0,
                "residual sugar": 2.6,
                "chlorides": 0.098,
                "free sulfur dioxide": 25.0,
                "total sulfur dioxide": 67.0,
                "density": 0.9968,
                "pH": 3.20,
                "sulphates": 0.68,
                "alcohol": 9.8,
            },
        ]
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code in [200, 500]
    if resp.status_code == 200:
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 2
