"""Verifying that a webhook really came from Twilio.

The endpoints in this service accept a phone call and can transfer it. Left
unauthenticated they are a free way for anyone on the internet to drive the
agent, burn model tokens and put calls into the escalation queue. The check is
fifteen lines, so there is no reason to skip it.

Twilio signs ``url + every POST parameter sorted by name and concatenated``
with the account auth token, HMAC-SHA1, base64. The URL must be exactly the
one Twilio requested, including scheme, host, port and query string -- which is
why ``PUBLIC_BASE_URL`` has to match the webhook configured in the console.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import urlencode


def expected_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    payload = url + "".join(
        key + str(params[key]) for key in sorted(params, key=str)
    )
    digest = hmac.new(
        auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def is_valid(
    auth_token: str, url: str, params: dict[str, str], signature: str
) -> bool:
    if not auth_token or not signature:
        return False
    return hmac.compare_digest(
        expected_signature(auth_token, url, params), signature
    )


def canonical_url(base_url: str, path: str, query: str = "") -> str:
    """Rebuild the URL Twilio signed.

    Behind a proxy or a tunnel, the URL the app sees is not the URL Twilio
    called -- the scheme is http, the host is an internal one, or the port is
    different. Signing against the configured public base URL instead removes
    a whole class of "signature invalid in production only" failures.
    """
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    return f"{url}?{query}" if query else url


def sign_for_testing(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Produce a valid signature. Used by the test suite and by curl recipes."""
    return expected_signature(auth_token, url, params)


def form_body(params: dict[str, str]) -> str:
    return urlencode(params)
