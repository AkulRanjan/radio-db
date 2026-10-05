# One image for both roles:
#   API:     uvicorn radiodb.api:app --host 0.0.0.0 --port 8000   (default CMD)
#   worker:  radiodb worker
#
# CPU build (default):  docker build -t radiodb .
# GPU build:            docker build -t radiodb:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 .
#                       (run with --gpus all and RADIODB_DEVICE=cuda)
FROM python:3.13-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RADIODB_HOME=/data

WORKDIR /app

# torch first, from the CPU or CUDA index, so the rest of the install reuses it.
RUN pip install torch torchaudio --index-url "${TORCH_INDEX}"

# Dependencies in their own layer (rebuilt only when pyproject.toml changes), then the code.
COPY pyproject.toml README.md ./
RUN mkdir radiodb && touch radiodb/__init__.py \
    && pip install ".[ml,postgres,s3]" \
    && pip uninstall -y radiodb && rm -rf radiodb
COPY radiodb ./radiodb
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 radiodb \
    && mkdir -p /data \
    && chown radiodb /data
USER radiodb
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "radiodb.api:app", "--host", "0.0.0.0", "--port", "8000"]
