"""Shared HTTP helpers.

Both the PubMed retriever and the LLM client need the same primitives:
  - bounded request timeouts
  - retry with exponential backoff on transient failures (429, 5xx,
    connection resets)
  - one consistent place to log every retry

Keeping this in one tiny module means we never have to remember which
client implemented retries and which did not.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


# HTTP status codes that we will retry. 429 is "too many requests" and
# anything in the 5xx range is a server-side blip.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Maximum number of attempts, including the first. With three attempts
# and an exponential backoff base of 0.5s we wait 0.5s then 1s between
# tries - low enough to feel responsive, high enough to recover from
# the bursts that NCBI and Groq occasionally throw.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.5


class HTTPRetryError(RuntimeError):
    """Raised when every retry has been exhausted."""


def request_with_retry(
    method: str,
    url: str,
    *,
    timeout: float,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Any = None,
) -> requests.Response:
    """Make an HTTP request and retry transient failures.

    The function returns the final :class:`requests.Response`. It does
    NOT raise on 4xx responses that are not in ``_RETRYABLE_STATUS``;
    those are returned to the caller so each client can decide how to
    interpret them.

    Raises:
        HTTPRetryError: every attempt failed.
    """
    last_exc: Optional[BaseException] = None
    last_status: Optional[int] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            last_status = None
            logger.warning(
                "HTTP %s %s attempt %d/%d failed with %s",
                method, url, attempt, max_attempts, exc,
            )
        else:
            if response.status_code not in _RETRYABLE_STATUS:
                return response
            last_status = response.status_code
            logger.warning(
                "HTTP %s %s attempt %d/%d returned %d - retrying",
                method, url, attempt, max_attempts, response.status_code,
            )

        if attempt < max_attempts:
            # Exponential backoff, capped at 8 seconds to keep UI snappy.
            delay = min(backoff_base * (2 ** (attempt - 1)), 8.0)
            time.sleep(delay)

    detail = f"status={last_status}" if last_status else f"exception={last_exc}"
    raise HTTPRetryError(
        f"All {max_attempts} attempts to {method} {url} failed ({detail})."
    )
