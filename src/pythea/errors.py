from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class TheaError(Exception):
    """Base error for pythea."""
    pass


@dataclass
class TheaHTTPError(TheaError):
    """Raised when the API returns a non-2xx response."""

    status_code: int
    message: str
    response_text: Optional[str] = None
    request_id: Optional[str] = None

    def __str__(self) -> str:
        rid = f" request_id={self.request_id}" if self.request_id else ""
        return f"TheaHTTPError(status_code={self.status_code}{rid}): {self.message}"


class TheaTimeoutError(TheaError):
    """Raised when an HTTP request times out."""
    pass


class TheaResponseError(TheaError):
    """Raised when a response cannot be parsed as expected."""
    pass
