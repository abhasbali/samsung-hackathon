# CodeFusion — CPU image: API + built frontend + CLI/evaluation scripts.
#   docker build -t codefusion .
#   docker run -p 8000:8000 -v codefusion-hf:/root/.cache/huggingface codefusion
# The first start indexes examples/sample_repo (the embedding model downloads on first use).

FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only PyTorch wheel first (avoids pulling CUDA libraries), then the rest.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md evaluate.py ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY examples/ examples/
RUN pip install --no-deps -e .
COPY --from=frontend /fe/dist frontend/dist
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENV CODEFUSION_CONFIG=/app/configs/full.yaml
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "codefusion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
