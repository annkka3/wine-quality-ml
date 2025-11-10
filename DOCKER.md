# Docker инструкции

## Сборка и запуск API

### С помощью Docker Compose (рекомендуется)

```bash
# Сборка и запуск
docker-compose up --build

# Запуск в фоне
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

### С помощью Docker напрямую

```bash
# Сборка образа
docker build -t wine-quality-api:latest .

# Запуск контейнера
docker run -d \
  --name wine-quality-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  wine-quality-api:latest

# Просмотр логов
docker logs -f wine-quality-api

# Остановка
docker stop wine-quality-api
docker rm wine-quality-api
```

## Важные замечания

1. **Модель должна быть доступна**: Перед запуском убедитесь, что файл `models/model_best.pkl` существует. Если модель версионируется через DVC, выполните `dvc pull` перед сборкой образа.

2. **Метаданные модели**: Файл `models/model_metadata.json` будет использоваться для endpoint `/model-info`, если доступен.

3. **Обновление модели**: Чтобы обновить модель без пересборки образа:
   ```bash
   # Обновите модель локально
   dvc pull models/model_best.pkl
   
   # Перезапустите контейнер
   docker-compose restart
   ```

## Проверка работы

После запуска API будет доступен по адресу: http://localhost:8000

- Healthcheck: http://localhost:8000/healthcheck
- API документация: http://localhost:8000/docs
- Информация о модели: http://localhost:8000/model-info

## Пример запроса

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
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
      "alcohol": 9.4
    }]
  }'
```

