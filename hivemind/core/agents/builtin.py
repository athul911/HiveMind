"""Built-in agent definitions provisioned at startup.

The SQL specialist is provisioned automatically if absent: specialized in PostgreSQL
query generation, schema introspection, optimization, and result interpretation; bound to
the SQL and code-execution tools and the postgres-optimization skill.
"""

from __future__ import annotations

from hivemind.config import Settings
from hivemind.core.agents.agent import Agent
from hivemind.core.llm.base import LLMConfig

SQL_SPECIALIST_NAME = "sql-specialist"

_SQL_SYSTEM_PROMPT = """\
You are HiveMind's PostgreSQL Specialist. You excel at:
- Translating natural-language questions into safe, parameterized SQL.
- Introspecting the database schema on demand before writing queries.
- Optimizing slow queries and explaining query plans.
- Interpreting results and summarizing them clearly for the user.

Rules:
- Use the `sql_query` tool. For schema discovery, call it with `introspect: true`.
- Emit exactly one read-only SELECT per query and pass values via bind parameters.
- When a computation or transformation is needed beyond SQL, use the `code_exec` tool.
- Prefer returning concise summaries over dumping large result sets.
"""


def sql_specialist_agent(settings: Settings) -> Agent:
    return Agent(
        name=SQL_SPECIALIST_NAME,
        description="PostgreSQL query generation, schema introspection, optimization.",
        system_prompt=_SQL_SYSTEM_PROMPT,
        tool_names=("sql_query", "code_exec"),
        skill_names=("postgres-optimization",),
        llm_config=LLMConfig(
            provider=settings.llm_default_provider,
            model=settings.llm_default_model,
            max_tokens=4096,
        ),
    )
