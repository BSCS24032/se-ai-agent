"""Tests for the shared HTTP retry helper."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from src.http_utils import (
    DEFAULT_MAX_ATTEMPTS,
    HTTPRetryError,
    request_with_retry,
)


class _Resp:
    def __init__(self, status: int, text: str = ""):
        self.status_code = status
        self.text = text

    def json(self):
        import json as _json

        return _json.loads(self.text) if self.text else {}


def test_returns_response_on_first_success():
    with patch(
        "src.http_utils.requests.request", return_value=_Resp(200, "{}")
    ) as mock_req:
        resp = request_with_retry("GET", "http://example.com", timeout=1)
    assert resp.status_code == 200
    assert mock_req.call_count == 1


def test_retries_on_429():
    responses = [_Resp(429), _Resp(429), _Resp(200, "{}")]
    with patch(
        "src.http_utils.requests.request", side_effect=responses
    ) as mock_req:
        with patch("src.http_utils.time.sleep") as mock_sleep:
            resp = request_with_retry(
                "GET", "http://example.com", timeout=1
            )
    assert resp.status_code == 200
    assert mock_req.call_count == 3
    assert mock_sleep.call_count == 2


def test_raises_after_max_attempts():
    with patch(
        "src.http_utils.requests.request", return_value=_Resp(503)
    ):
        with patch("src.http_utils.time.sleep"):
            with pytest.raises(HTTPRetryError):
                request_with_retry(
                    "GET",
                    "http://example.com",
                    timeout=1,
                    max_attempts=DEFAULT_MAX_ATTEMPTS,
                )


def test_does_not_retry_on_404():
    """Non-retryable status codes should be returned to the caller."""
    with patch(
        "src.http_utils.requests.request", return_value=_Resp(404)
    ) as mock_req:
        resp = request_with_retry("GET", "http://example.com", timeout=1)
    assert resp.status_code == 404
    assert mock_req.call_count == 1


def test_retries_on_connection_exception():
    side_effect = [
        requests.ConnectionError("boom"),
        _Resp(200, "{}"),
    ]
    with patch("src.http_utils.requests.request", side_effect=side_effect):
        with patch("src.http_utils.time.sleep"):
            resp = request_with_retry(
                "GET", "http://example.com", timeout=1
            )
    assert resp.status_code == 200
