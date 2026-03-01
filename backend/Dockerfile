FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Artifacts directory — mount at runtime or bake in with COPY
RUN mkdir -p /app/artifacts

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV ARTIFACTS_DIR=/app/artifacts
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
