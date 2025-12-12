# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libasound2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy your service account JSON (path in repo: arcane-attic-467611-b7-89fc8db14f77.json)
COPY arcane-attic-467611-b7-89fc8db14f77.json /app/creds/service-account.json

# Copy the rest of the code
COPY . .

EXPOSE 8000
ENV ENVIRONMENT=production DEBUG=false LOG_LEVEL=INFO

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
