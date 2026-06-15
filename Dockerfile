# syntax=docker/dockerfile:1
# Multi-stage build using uv for fast, reproducible installs.
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# uv for dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ---- dependency layer (cached) ----
FROM base AS deps
COPY pyproject.toml ./
RUN uv pip install --system --no-cache .

# ---- runtime ----
FROM deps AS runtime
# bubblewrap powers the subprocess backend's `namespaces` isolation (SUBPROCESS_ISOLATION).
# Unused unless enabled; relies on unprivileged user namespaces at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*
COPY hivemind ./hivemind
COPY skills ./skills
COPY alembic.ini ./

# Non-root runtime user.
RUN useradd --uid 10001 --create-home appuser \
    && mkdir -p /data/artifacts && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
# Default command runs the API; the worker overrides this in compose/Helm.
CMD ["uvicorn", "hivemind.main:app", "--host", "0.0.0.0", "--port", "8000"]
