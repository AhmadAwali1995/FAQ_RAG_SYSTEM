"""Shared HTTP helpers for the provider clients.

Both the embedding and generation backends talk to hosted APIs over httpx, and
both need to report a failure the same way.
"""

from __future__ import annotations

import httpx

#: Provider error bodies are quoted in exceptions; cap them so a stack trace
#: stays readable when an endpoint returns an HTML error page.
_MAX_DETAIL_CHARS = 500


def describe_http_error(exc: httpx.HTTPError) -> str:
    """Render an httpx error, including the provider's own message when present.

    A hosted provider explains a rejected request in the response body -- an
    unknown model ID, a revoked key, a token budget over the model's cap.
    Without the body a 400 reads only as "Bad Request", which sends you hunting
    the prompt for a fault that is really a config value.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        detail = exc.response.text.strip()
        if detail:
            return f"{exc.response.status_code} {detail[:_MAX_DETAIL_CHARS]}"
    return str(exc)
