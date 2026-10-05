"""Who is calling the API.

Modes (``RADIODB_AUTH``):
  * ``header`` (default; local development only): trust the ``X-User-Id`` header.
  * ``jwt``: require ``Authorization: Bearer <token>`` and verify it with either a shared
    secret (``RADIODB_JWT_SECRET``, HS256) or the identity provider's public keys
    (``RADIODB_JWT_JWKS_URL``, RS256/ES256: Auth0, Cognito, Keycloak, Entra ID, ...).
    ``exp`` is always required; ``aud``/``iss`` are checked when configured. The user id is
    the ``RADIODB_JWT_USER_CLAIM`` claim (default ``sub``).
"""

from __future__ import annotations

import logging
from functools import cached_property

from .config import Settings

log = logging.getLogger(__name__)


class AuthError(Exception):
    pass


class Authenticator:
    def __init__(self, settings: Settings):
        self.s = settings
        if settings.auth_mode not in ("header", "jwt"):
            raise ValueError(f"RADIODB_AUTH must be 'header' or 'jwt', not {settings.auth_mode!r}")
        if settings.auth_mode == "jwt" and not (settings.jwt_secret or settings.jwt_jwks_url):
            raise ValueError("RADIODB_AUTH=jwt needs RADIODB_JWT_SECRET or RADIODB_JWT_JWKS_URL")
        if settings.auth_mode == "header":
            log.warning("RADIODB_AUTH=header: trusting the X-User-Id header. Local development only.")

    @cached_property
    def _jwks(self):
        from jwt import PyJWKClient

        return PyJWKClient(self.s.jwt_jwks_url, cache_keys=True, lifespan=3600)

    def user_id(self, authorization: str | None, x_user_id: str | None) -> str:
        if self.s.auth_mode == "header":
            if not x_user_id or len(x_user_id) > 128:
                raise AuthError("missing or invalid X-User-Id header")
            return x_user_id
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("missing bearer token")
        return self._verify(authorization[7:].strip())

    def _verify(self, token: str) -> str:
        import jwt

        try:
            if self.s.jwt_jwks_url:
                key = self._jwks.get_signing_key_from_jwt(token).key
                algorithms = ["RS256", "RS384", "RS512", "ES256", "ES384", "PS256"]
            else:
                key, algorithms = self.s.jwt_secret, ["HS256", "HS384", "HS512"]
            claims = jwt.decode(
                token, key, algorithms=algorithms,
                audience=self.s.jwt_audience or None, issuer=self.s.jwt_issuer or None,
                options={"require": ["exp"], "verify_aud": bool(self.s.jwt_audience)},
                leeway=30,
            )
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid token: {e}") from e
        user = claims.get(self.s.jwt_user_claim)
        if not isinstance(user, str) or not user or len(user) > 128:
            raise AuthError(f"token has no usable '{self.s.jwt_user_claim}' claim")
        return user
