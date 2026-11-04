# On an x86_64 Linux host the PyPI torch wheel bundles CUDA, so `docker run --gpus all` trains on GPU.
# On other platforms (for example an arm64 Mac) the same build gives a CPU-only torch.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends make git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8 MLFLOW_DISABLE_AGENT_HINT=1
CMD ["make", "reproduce", "PY=python"]
