"""Artifact download endpoint.

Owner-authenticated: the caller must present their bearer token (verified like any other
endpoint — JWKS/RS256 or HS256), and the server checks they own the artifact. The id in the
URL only *names* the artifact (it encodes the base-relative path); it is not a credential.
The path is jailed under the artifact base directory, and ownership is derived server-side
from the artifact's conversation — never trusted from the id itself.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from hivemind.api.authz import assert_conversation_access
from hivemind.api.deps import AppCtx, CurrentUser
from hivemind.core.errors import AuthorizationError, NotFoundError
from hivemind.db.repository import ConversationRepository

router = APIRouter(tags=["artifacts"])


@router.get("/v1/artifacts/{artifact_id}")
async def download_artifact(artifact_id: str, app: AppCtx, user: CurrentUser):
    artifacts = app.artifacts
    try:
        path = artifacts.resolve_id(artifact_id)
    except ValueError as exc:
        raise NotFoundError("Artifact not found.") from exc

    # Ownership: derive the artifact's conversation and require the caller to own it. We deny
    # rather than 404 when ownership can't be established, so an authenticated user can't probe
    # for files outside any conversation they own.
    conversation_id = artifacts.owner_conversation_id(path)
    convo = None
    if conversation_id is not None:
        async with app.db.session() as session:
            convo = await ConversationRepository(session).get(conversation_id)
    if convo is None:
        raise AuthorizationError("You do not have access to this artifact.")
    assert_conversation_access(convo, user, app.settings)

    if not path.is_file():
        raise NotFoundError("Artifact not found or no longer available.")

    return FileResponse(
        Path(path),
        filename=path.name,
        media_type=artifacts.describe(path).mime_type,
    )
