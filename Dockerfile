# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости (если нужны)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY src/ ./src/
COPY models/ ./models/

# Создаем директорию для моделей, если её нет
RUN mkdir -p models

# Открываем порт для API
EXPOSE 8000

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Команда для запуска API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

