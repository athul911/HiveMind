from __future__ import annotations

import pytest
from hivemind.config import Settings
from hivemind.core.errors import UnsafeSQLError
from hivemind.core.tools.sql_tool import SQLTool
from hivemind.services.artifact_store import ArtifactStore


@pytest.fixture
def tool(tmp_path) -> SQLTool:
    settings = Settings(database_url="postgresql+asyncpg://x/y")
    return SQLTool(settings, ArtifactStore(str(tmp_path)))


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "SELECT 1; DROP TABLE t",
        "SELECT 1; SELECT 2",
        "CREATE TABLE t (id int)",
        "TRUNCATE t",
    ],
)
def test_unsafe_sql_rejected(tool: SQLTool, sql: str):
    with pytest.raises(UnsafeSQLError):
        tool._guard(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM demo_sales",
        "SELECT id, product FROM demo_sales WHERE region = :region",
        "WITH r AS (SELECT 1) SELECT * FROM r",
        "select count(*) from demo_sales;",
    ],
)
def test_safe_sql_allowed(tool: SQLTool, sql: str):
    # Should not raise.
    tool._guard(sql)
