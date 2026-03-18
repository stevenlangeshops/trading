# Debian Bookworm (12) – stabil, libatlas verfügbar
FROM python:3.11-slim-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libhdf5-dev libatlas-base-dev gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM deps AS app
COPY . .
RUN mkdir -p data/raw features/processed checkpoints logs
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "main.py"]
CMD ["all", "--symbol", "BTC/USDT", "--timeframe", "1h", "--threshold", "0.002"]
