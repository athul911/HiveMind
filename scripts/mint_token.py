"""Mint a short-lived HS256 JWT for local API testing.

Usage:
    python scripts/mint_token.py [user_id]

Prints a Bearer token signed with JWT_SECRET from the environment/.env.
"""

from __future__ import annotations

import sys
import time
import warnings

import jwt

from hivemind.config import get_settings

# This is a local-dev helper; a short default JWT_SECRET is fine here. Quiet the SDK's
# key-length advisory so the token is the only thing printed. (Use a 32+ byte secret
# in production — see .env.example.)
warnings.filterwarnings("ignore", message=".*HMAC key.*", module="jwt")


def main() -> None:
    settings = get_settings()
    user_id = sys.argv[1] if len(sys.argv) > 1 else "dev-user"
    claims = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    if settings.jwt_issuer:
        claims["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        claims["aud"] = settings.jwt_audience
    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    print(token)


if __name__ == "__main__":
    main()
