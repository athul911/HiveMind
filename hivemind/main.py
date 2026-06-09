"""FastAPI application factory + lifespan.

The lifespan builds the full :class:`AppContext` (composition root), connects the broker and
Redis, starts the cleanup scheduler, and instruments the app. On SIGTERM the lifespan tears
everything down — and uvicorn drains in-flight requests first (graceful shutdown).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hivemind.api.auth import TokenVerifier
from hivemind.api.errors import install_exception_handlers
from hivemind.api.middleware.context_middleware import ContextMiddleware
from hivemind.api.middleware.ratelimit import RateLimitMiddleware
from hivemind.api.routes import agents, catalog, chat, health, tasks
from hivemind.bootstrap import build_context
from hivemind.config import Settings, get_settings
from hivemind.observability.logging import get_logger
from hivemind.observability.tracing import instrument_app
from hivemind.workers.broker import TaskBroker
from hivemind.workers.dispatcher import TaskDispatcher
from hivemind.workers.events import TaskEventBuffer

logger = get_logger("hivemind.main")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with build_context(settings) as ctx:
            redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            broker = TaskBroker(settings.rabbitmq_url, settings.rabbitmq_task_queue)
            with contextlib.suppress(Exception):
                await broker.connect()

            app.state.context = ctx
            app.state.redis = redis
            app.state.broker = broker
            app.state.dispatcher = TaskDispatcher(ctx.db, broker)
            app.state.event_buffer = TaskEventBuffer(ctx.db, redis)

            ctx.scheduler.start()
            instrument_app(app, engine=ctx.db.engine)
            logger.info("api.started", environment=settings.environment)
            try:
                yield
            finally:
                with contextlib.suppress(Exception):
                    await broker.close()
                with contextlib.suppress(Exception):
                    await redis.aclose()

    app = FastAPI(
        title="HiveMind",
        version="0.1.0",
        description="Multi-agent AI orchestration on LangGraph.",
        lifespan=lifespan,
    )

    verifier = TokenVerifier(settings)
    # Middleware executes in reverse registration order: CORS outermost, then auth/context,
    # then rate-limit (which needs the authenticated principal).
    redis_for_rl = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.add_middleware(
        RateLimitMiddleware, redis=redis_for_rl, per_minute=settings.rate_limit_per_minute
    )
    app.add_middleware(ContextMiddleware, verifier=verifier)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(agents.router)
    app.include_router(catalog.router)
    app.include_router(tasks.router)
    _install_bearer_security(app)
    return app


def _install_bearer_security(app: FastAPI) -> None:
    """Advertise Bearer auth in the OpenAPI schema so Swagger shows an Authorize button.

    Auth is enforced in middleware (not via route dependencies), so FastAPI wouldn't
    otherwise know to render the token field. We inject the security scheme into the
    generated schema and apply it globally.
    """
    from fastapi.openapi.utils import get_openapi

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        # Global default; public endpoints (health/metrics) ignore it harmlessly.
        schema["security"] = [{"bearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi


app = create_app()
