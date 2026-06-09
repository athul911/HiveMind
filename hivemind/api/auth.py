"""JWT verification: HS256 shared-secret (default) or OAuth2 JWKS/RS256 (optional).

When ``OAUTH2_JWKS_URL`` is configured we verify RS256 tokens against the cached JWKS
(selected by ``kid``); otherwise we verify HS256 with the shared secret. The verified
``sub`` claim becomes the request's ``user_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from jwt import PyJWKClient

from hivemind.config import Settings
from hivemind.core.errors import AuthenticationError


@dataclass(frozen=True)
class Principal:
    user_id: str
    claims: dict


class TokenVerifier:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jwks_client: PyJWKClient | None = (
            PyJWKClient(settings.oauth2_jwks_url) if settings.oauth2_jwks_url else None
        )

    def verify(self, token: str) -> Principal:
        if self._settings.auth_disabled:
            return Principal(user_id="local-dev", claims={"sub": "local-dev"})
        try:
            claims = self._decode(token)
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"Invalid token: {exc}") from exc
        sub = claims.get("sub")
        if not sub:
            raise AuthenticationError("Token missing 'sub' claim.")
        return Principal(user_id=str(sub), claims=claims)

    def _decode(self, token: str) -> dict:
        options = {"verify_aud": self._settings.jwt_audience is not None}
        common = {
            "audience": self._settings.jwt_audience,
            "issuer": self._settings.jwt_issuer,
            "options": options,
        }
        if self._jwks_client is not None:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(token, signing_key.key, algorithms=["RS256"], **common)
        return jwt.decode(
            token,
            self._settings.jwt_secret,
            algorithms=[self._settings.jwt_algorithm],
            **common,
        )
