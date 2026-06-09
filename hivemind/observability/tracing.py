"""OpenTelemetry setup: tracing + metrics, custom spans, and HiveMind metrics.

Exports OTLP to a configurable collector and exposes a Prometheus-compatible metrics
reader. Custom spans wrap graph-node execution, tool invocations, LLM calls, and routing
decisions. Custom metrics: ``hivemind.workflow.duration``, ``hivemind.tool.calls.total``,
``hivemind.llm.tokens.used``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.trace import Span, Status, StatusCode

from hivemind.config import Settings
from hivemind.core.context import get_context

_initialized = False
_workflow_duration: Histogram | None = None
_tool_calls: Counter | None = None
_llm_tokens: Counter | None = None


def setup_telemetry(settings: Settings) -> None:
    """Initialize tracer and meter providers. Idempotent."""
    global _initialized, _workflow_duration, _tool_calls, _llm_tokens
    if _initialized or not settings.otel_enabled:
        _initialized = True
        return

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({SERVICE_NAME: settings.service_name})

    tracer_provider = TracerProvider(resource=resource)
    metric_readers: list[Any] = []

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
            )
        )

    # Prometheus reader for K8s-side scraping.
    with contextlib.suppress(Exception):
        from opentelemetry.exporter.prometheus import PrometheusMetricReader

        metric_readers.append(PrometheusMetricReader())

    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=metric_readers))

    meter = metrics.get_meter("hivemind")
    _workflow_duration = meter.create_histogram(
        "hivemind.workflow.duration", unit="s", description="End-to-end workflow duration."
    )
    _tool_calls = meter.create_counter(
        "hivemind.tool.calls.total", description="Total tool invocations."
    )
    _llm_tokens = meter.create_counter(
        "hivemind.llm.tokens.used", description="LLM tokens consumed."
    )
    _initialized = True


def instrument_app(app: Any, engine: Any = None) -> None:
    """Apply FastAPI / SQLAlchemy / HTTPX auto-instrumentation."""
    with contextlib.suppress(Exception):
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    with contextlib.suppress(Exception):
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    if engine is not None:
        with contextlib.suppress(Exception):
            from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

            SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)


def get_tracer(name: str = "hivemind") -> trace.Tracer:
    return trace.get_tracer(name)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Open a custom span enriched with request-context attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as sp:
        ctx = get_context()
        if ctx is not None:
            for key, value in ctx.to_log_dict().items():
                sp.set_attribute(f"hivemind.{key}", value)
        for key, value in attributes.items():
            if value is not None:
                sp.set_attribute(key, value)
        try:
            yield sp
        except Exception as exc:
            sp.set_status(Status(StatusCode.ERROR, str(exc)))
            sp.record_exception(exc)
            raise


def record_tool_call(tool_name: str, *, success: bool) -> None:
    if _tool_calls is not None:
        _tool_calls.add(1, {"tool": tool_name, "success": str(success).lower()})


def record_llm_tokens(provider: str, model: str, *, input_tokens: int, output_tokens: int) -> None:
    if _llm_tokens is not None:
        attrs = {"provider": provider, "model": model}
        _llm_tokens.add(input_tokens, {**attrs, "direction": "input"})
        _llm_tokens.add(output_tokens, {**attrs, "direction": "output"})


def record_workflow_duration(seconds: float, *, mode: str) -> None:
    if _workflow_duration is not None:
        _workflow_duration.record(seconds, {"mode": mode})
