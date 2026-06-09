"""Liveness/readiness/metrics endpoints (public, no auth)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request, Response
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/healthz")
async def health():
    return {"status": "ok"}


async def _check_postgres(app) -> bool:
    async with app.db.session() as session:
        await session.execute(text("SELECT 1"))
    return True


async def _check_redis(request: Request) -> bool:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return True  # not configured in this process
    await redis.ping()
    return True


async def _check_rabbitmq(request: Request) -> bool:
    broker = getattr(request.app.state, "broker", None)
    conn = getattr(broker, "_connection", None) if broker is not None else None
    if conn is None:
        return True  # broker optional (e.g. SSE-only deployment)
    return not conn.is_closed


@router.get("/readyz")
async def readyz(request: Request):
    """Ready only when every backing dependency this process needs is reachable."""
    app = request.app.state.context
    checks = {"postgres": _check_postgres(app), "redis": _check_redis(request)}
    checks["rabbitmq"] = _check_rabbitmq(request)

    results: dict[str, str] = {}
    healthy = True
    for name, coro in checks.items():
        try:
            await coro
            results[name] = "ok"
        except Exception:
            results[name] = "error"
            healthy = False

    status = "ready" if healthy else "not-ready"
    body = json.dumps({"status": status, "checks": results})
    return Response(
        content=body,
        status_code=200 if healthy else 503,
        media_type="application/json",
    )


@router.get("/metrics")
async def metrics():
    """Prometheus exposition (OTel Prometheus reader registers collectors globally)."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
