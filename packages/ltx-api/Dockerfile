FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY packages/ltx-core/ packages/ltx-core/
COPY packages/ltx-pipelines/ packages/ltx-pipelines/
COPY packages/ltx-api/ packages/ltx-api/
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --extra ltx-api

ENV LTX_API_HOST=0.0.0.0
ENV LTX_API_PORT=8000

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--no-dev", "python", "-m", "ltx_api"]
