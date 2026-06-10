"""Chat (OpenAI-compatible) request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request, extended with HiveMind fields."""

    messages: list[ChatMessage]
    model: str | None = None
    stream: bool = True
    conversation_id: str | None = Field(
        default=None, description="Reuse an existing conversation; created if omitted."
    )
    agent_id: str | None = Field(
        default=None, description="Pin a specific agent; otherwise the supervisor routes."
    )

    @property
    def latest_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return self.messages[-1].content if self.messages else ""


class AsyncTaskResponse(BaseModel):
    task_id: str
    conversation_id: str
    status: str = "queued"
    stream_url: str
    status_url: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
    error: str | None = None
    usage: dict = Field(default_factory=dict)


class TaskSummary(BaseModel):
    """A task in a conversation's history, with links to reconnect to it."""

    task_id: str
    status: str
    created_at: str
    completed_at: str | None = None
    stream_url: str
    status_url: str


class ConversationSummary(BaseModel):
    """A conversation owned by the current user."""

    id: str
    status: str
    agent_id: str | None = None
    created_at: str
    updated_at: str
