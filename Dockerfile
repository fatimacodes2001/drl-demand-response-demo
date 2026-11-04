FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
CMD ["make", "reproduce", "PY=python"]
