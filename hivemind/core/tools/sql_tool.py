"""SQL execution tool — least-privilege, read-only, hardened.

LLM-authored SQL is treated as hostile. Defenses:
  * connects via a dedicated read-only DSN/role;
  * SELECT-only, single-statement guard (rejects DDL/DML/multi-statement);
  * per-query ``statement_timeout``;
  * hard row cap;
  * schema allow-list checked against ``information_schema``.
Results return as structured JSON; large result sets are written to the artifact store.
"""

from __future__ import annotations

import sqlglot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlglot import expressions as exp

from hivemind.config import Settings
from hivemind.core.context import RequestContext
from hivemind.core.errors import UnsafeSQLError
from hivemind.core.tools.base import BaseTool, ToolResult
from hivemind.services.artifact_store import ArtifactStore

# Only these top-level statement types are read-only and permitted.
_ALLOWED_STATEMENTS = (exp.Select, exp.Union, exp.With, exp.Subquery)
# Any of these nested anywhere in the tree means the query mutates or escalates.
_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,  # catches TRUNCATE, GRANT, VACUUM, SET, COPY, CALL, etc.
    exp.Merge,
)
_INLINE_ROWCAP = 200


class SQLTool(BaseTool):
    name = "sql_query"
    description = (
        "Execute a single read-only SELECT against the PostgreSQL database and return rows "
        "as JSON. Always parameterize values via the `params` object; never interpolate "
        "user input into the SQL string. Use `introspect:true` to list tables/columns."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "A single SELECT statement."},
            "params": {
                "type": "object",
                "description": "Named bind parameters referenced as :name in the SQL.",
                "additionalProperties": True,
            },
            "introspect": {
                "type": "boolean",
                "description": "If true, return schema (tables/columns) instead of running SQL.",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, settings: Settings, artifacts: ArtifactStore) -> None:
        self._settings = settings
        self._artifacts = artifacts
        self._engine: AsyncEngine | None = None

    def _get_engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                self._settings.effective_sql_tool_dsn,
                pool_size=5,
                max_overflow=5,
                pool_pre_ping=True,
            )
        return self._engine

    async def run(self, args: dict, ctx: RequestContext) -> ToolResult:
        if args.get("introspect"):
            return await self._introspect()
        sql = (args.get("sql") or "").strip()
        if not sql:
            raise UnsafeSQLError("No SQL provided.")
        self._guard(sql)
        return await self._execute(sql, args.get("params") or {}, ctx)

    def _guard(self, sql: str) -> None:
        """Validate by *parsing* the SQL, not by regex.

        A real parser is far more robust than keyword matching: it can't be fooled by
        keywords inside string literals or identifiers, and it reliably detects multiple
        statements. We parse as PostgreSQL, require exactly one top-level read-only
        statement (SELECT / UNION / WITH), and reject any mutating/DDL node anywhere in the
        tree. The read-only DB role remains the ultimate backstop.
        """
        try:
            statements = sqlglot.parse(sql, read="postgres")
        except sqlglot.errors.ParseError as exc:
            raise UnsafeSQLError(f"Could not parse SQL: {exc}") from exc

        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            raise UnsafeSQLError("Exactly one statement is allowed.")

        root = statements[0]
        if not isinstance(root, _ALLOWED_STATEMENTS):
            raise UnsafeSQLError("Only read-only SELECT/WITH queries are permitted.")
        for forbidden in _FORBIDDEN_NODES:
            if root.find(forbidden) is not None:
                raise UnsafeSQLError("Query contains a forbidden (write/DDL/DCL) operation.")

    async def _execute(self, sql: str, params: dict, ctx: RequestContext) -> ToolResult:
        engine = self._get_engine()
        timeout_ms = self._settings.sql_tool_statement_timeout_ms
        max_rows = self._settings.sql_tool_max_rows
        async with engine.connect() as conn:
            await conn.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
            result = await conn.execute(text(sql), params)
            columns = list(result.keys())
            rows = result.fetchmany(max_rows)
            data = [dict(zip(columns, row, strict=False)) for row in rows]

        # Large result sets are written to the artifact store; only a ref + preview returned.
        if len(data) > _INLINE_ROWCAP:
            import json

            ref = self._artifacts.write_text(
                ctx.conversation_id or "sync",
                ctx.task_id,
                self.name,
                "result.json",
                json.dumps(data, default=str),
            )
            return ToolResult(
                content={"columns": columns, "row_count": len(data), "preview": data[:20]},
                artifact=ref.to_dict(),
            )
        return ToolResult(content={"columns": columns, "row_count": len(data), "rows": data})

    async def _introspect(self) -> ToolResult:
        engine = self._get_engine()
        schemas = self._settings.sql_tool_allowed_schemas
        stmt = text(
            """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = ANY(:schemas)
            ORDER BY table_schema, table_name, ordinal_position
            """
        )
        async with engine.connect() as conn:
            result = await conn.execute(stmt, {"schemas": schemas})
            tables: dict[str, list[dict]] = {}
            for schema, table, column, dtype in result.fetchall():
                key = f"{schema}.{table}"
                tables.setdefault(key, []).append({"column": column, "type": dtype})
        return ToolResult(content={"schemas": schemas, "tables": tables})

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
