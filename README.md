# Wine Quality ML

Учебный ML-проект по предсказанию **качества красного вина** из набора физико-химических признаков.

Проект собран в формате мини-MLOps-пайплайна:

- версионирование **данных и модели** через DVC;
- эксперименты и трекинг метрик в **MLflow**;
- автоматизация обучения в **Airflow**;
- REST **API на FastAPI** для инференса;
- **GitLab CI/CD** для линтинга, проверки DVC, тестов, сборки и деплоя Docker образа;
- **Docker** для контейнеризации API;
- **Метаданные модели** для версионирования и мониторинга;
- **Кросс-валидация и GridSearchCV** для подбора гиперпараметров;
- **Мониторинг деградации модели** с автоматическими алертами.

---

## 1. Постановка задачи

- **Датасет**: [`winequality-red.csv`](https://github.com/aniruddhachoudhury/Red-Wine-Quality/blob/master/winequality-red.csv)  
- **Цель**: предсказать численное качество вина (переменная `quality`) по набору признаков:
  - `fixed acidity`, `volatile acidity`, `citric acid`,  
  - `residual sugar`, `chlorides`,  
  - `free sulfur dioxide`, `total sulfur dioxide`,  
  - `density`, `pH`, `sulphates`, `alcohol`.

Тип задачи: **регрессия** (качество как число).

---

## 2. Стек технологий

- **Python** 3.11
- **pandas**, **numpy**, **scikit-learn**
- **DVC** — версионирование данных и модели
- **MLflow** — трекинг экспериментов
- **FastAPI** + **Uvicorn** — API-сервис
- **Apache Airflow** — DAG для обучения
- **pytest** — тесты
- **GitLab CI** — CI-пайплайн

---

## 3. Структура проекта

```text
wine-quality-ml/
  README.md
  requirements.txt
  params.yaml              # гиперпараметры моделей и train/test split

  data/
    winequality-red.csv.dvc  # данные под DVC

  models/
    model_best.pkl.dvc     # лучшая модель под DVC (файл .pkl — через dvc pull)
    model_metadata.json.dvc # метаданные модели (метрики, дата обучения, версия)
    metrics_history.json    # история метрик всех выбранных моделей

  mlruns/                  # эксперименты MLflow (локальный файловый backend)
  dvc_storage/             # локальный remote для DVC (по умолчанию)

  dags/
    wine_train_dag.py      # DAG для Airflow

  src/
    __init__.py

    data/
      __init__.py          # (зарезервировано под доп. data-пайплайны)

    models/
      __init__.py
      train_baseline.py    # простой baseline RandomForest без MLflow
      train.py             # основной train-скрипт с MLflow и двумя моделями
      select_best.py       # выбор лучшей модели из MLflow и сохранение метаданных

    api/
      __init__.py
      schemas.py           # Pydantic-схемы запросов/ответов
      model_loader.py      # загрузка model_best.pkl
      main.py              # FastAPI-приложение (эндпоинты /predict, /healthcheck, /model-info)

  tests/
    test_api.py            # тесты API (healthcheck + predict)

  .gitignore
  .dockerignore            # файлы, исключаемые из Docker образа
  Dockerfile               # Docker образ для API
  docker-compose.yml       # Docker Compose конфигурация
  DOCKER.md                # инструкции по работе с Docker
  .dvc/
  .dvc/config              # конфигурация DVC, remote storage
  .gitlab-ci.yml           # GitLab CI pipeline


## 4. Установка и настройка

### Требования

- **Python** 3.11
- **Git**
- **DVC** (устанавливается через pip)

### Установка зависимостей

```bash
# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

### Версии пакетов

Все зависимости зафиксированы в `requirements.txt` с указанием диапазонов версий для обеспечения совместимости:
- **pandas** >=2.0.0,<3.0.0
- **numpy** >=1.24.0,<2.0.0
- **scikit-learn** >=1.3.0,<2.0.0
- **mlflow** >=2.8.0,<3.0.0
- **dvc** >=3.0.0,<4.0.0
- **fastapi** >=0.104.0,<1.0.0
- **uvicorn** >=0.24.0,<1.0.0
- **pytest** >=7.4.0,<9.0.0
- **httpx** >=0.25.0,<1.0.0
- **pyyaml** >=6.0.1,<7.0.0

---

## 5. Данные и DVC

### Источник данных

Используется датасет `winequality-red.csv`:

- оригинальный источник: GitHub (см. ссылку в постановке задачи);
- в репозитории **сам CSV не хранится**, только `.dvc`-файл.

### Настройка DVC

Инициализация (однократно):

```bash
dvc init
dvc remote add -d storage ./dvc_storage
git add .dvc .gitignore
git commit -m "Init DVC"
```

---

## 6. Метаданные модели

При выборе лучшей модели (`select_best.py`) автоматически создается файл `models/model_metadata.json` с полной информацией о модели:

- **model_name**: тип модели (RandomForestRegressor/GradientBoostingRegressor)
- **run_id**: ID эксперимента в MLflow
- **version**: версия модели
- **trained_at**: дата и время обучения
- **selected_at**: дата и время выбора как лучшей
- **metrics**: метрики (RMSE, MAE, MSE)
- **parameters**: гиперпараметры модели
- **selection_criterion**: критерий выбора (lowest_rmse)

Метаданные используются в API endpoint `/model-info` для отображения актуальной информации о модели. Файл версионируется через DVC вместе с моделью.

---

## 7. Docker

API сервис упакован в Docker контейнер для удобного деплоя и изоляции окружения.

### Быстрый старт

```bash
# С помощью Docker Compose
docker-compose up --build

# Или напрямую через Docker
docker build -t wine-quality-api:latest .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro wine-quality-api:latest
```

Подробные инструкции см. в [DOCKER.md](DOCKER.md).

### Важно

Перед запуском контейнера убедитесь, что модель доступна:
```bash
dvc pull models/model_best.pkl models/model_metadata.json
```

---

## 8. Обучение моделей с кросс-валидацией и GridSearch

Проект поддерживает автоматический подбор гиперпараметров с помощью GridSearchCV и кросс-валидации.

### Использование

```bash
# Обучение с GridSearchCV (по умолчанию)
python -m src.models.train --model random_forest
python -m src.models.train --model gbrt

# Обучение без GridSearch (с фиксированными параметрами)
python -m src.models.train --model random_forest --no-grid-search
```

### Настройка параметров GridSearch

Параметры для GridSearch настраиваются в `params.yaml`:

```yaml
models:
  random_forest:
    grid_search:
      n_estimators: [100, 150, 200]
      max_depth: [6, 8, 10, None]
      min_samples_split: [2, 5, 10]
  gbrt:
    grid_search:
      n_estimators: [200, 300, 400]
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [3, 5, 7]

train:
  cv_folds: 5  # Количество фолдов для кросс-валидации
```

### Метрики

При обучении логируются следующие метрики:
- **CV метрики**: среднее и стандартное отклонение RMSE по фолдам кросс-валидации
- **Test метрики**: RMSE, MAE, MSE на тестовом наборе
- **Best parameters**: лучшие гиперпараметры, найденные GridSearchCV

---

## 9. Мониторинг и детекция деградации модели

При выборе лучшей модели (`select_best.py`) автоматически выполняется проверка на деградацию:

### Функции мониторинга

1. **Сравнение с предыдущей моделью**: сравнение метрик текущей и предыдущей лучшей модели
2. **Детекция деградации**: автоматическое определение ухудшения метрик (порог: 5% для RMSE)
3. **История метрик**: сохранение истории всех выбранных моделей в `models/metrics_history.json`
4. **Алерты**: предупреждения при обнаружении деградации

### Пример вывода

```
=== Model Comparison ===
Previous RMSE: 0.6234
Current RMSE:  0.6543
⚠️  Degradation: 4.96%
```

При превышении порога:
```
🚨 WARNING: Model degradation exceeds threshold (5.0%)!
   Consider reviewing the training process or data quality.
```

### Настройка порога деградации

Порог можно изменить в `src/models/select_best.py`:
```python
DEGRADATION_THRESHOLD_RMSE = 5.0  # 5% ухудшение RMSE
```

---

## 10. CI/CD Pipeline

GitLab CI/CD автоматизирует весь процесс разработки и деплоя:

### Стадии пайплайна

1. **Lint**: проверка кода с помощью flake8
2. **DVC Check**: проверка наличия данных и модели
3. **Tests**: запуск тестов API
4. **Build**: сборка Docker образа и публикация в GitLab Container Registry
5. **Deploy**: деплой в staging/production (ручной запуск)

### Docker образы

Образы автоматически публикуются в GitLab Container Registry с тегами:
- `$CI_COMMIT_SHA` - уникальный тег для каждого коммита
- `$CI_COMMIT_REF_SLUG` - тег для ветки (main, develop, etc.)
- `latest` - последняя версия

### Деплой

Деплой выполняется вручную через GitLab UI:
- **Staging**: для ветки `develop`
- **Production**: для веток `main`/`master`

Для настройки реального деплоя раскомментируйте и настройте команды в `.gitlab-ci.yml`:
```yaml
# Пример для Kubernetes:
# - kubectl set image deployment/wine-quality-api api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

# Пример для Docker Compose:
# - docker-compose pull && docker-compose up -d
```

---
