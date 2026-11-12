# Backend image. The scientific core is pure numpy/networkx, so the default
# build needs neither CUDA nor GDAL and stays small. Torch and rasterio are
# opt-in build args for deployments that actually run inference or GeoTIFFs.
FROM python:3.11-slim AS base

ARG WITH_TORCH=false
ARG WITH_GEO=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libGL/libglib are needed by opencv even in headless form.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-geo.txt requirements-dl.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install fastapi "uvicorn[standard]" pydantic python-multipart \
 && if [ "$WITH_GEO" = "true" ]; then pip install -r requirements-geo.txt; fi \
 && if [ "$WITH_TORCH" = "true" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
      && pip install transformers timm segmentation-models-pytorch; fi

COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Run as a non-root user: uploads are untrusted input.
RUN useradd -m -u 10001 appuser && mkdir -p data/uploads checkpoints \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
