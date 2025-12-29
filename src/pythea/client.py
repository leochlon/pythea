from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Union

import httpx

from ._version import __version__
from .errors import TheaHTTPError, TheaResponseError, TheaTimeoutError
from .types import BackendCredentials, UnifiedAnswerRequest, UnifiedAnswerResponse
from .utils import merge_headers, normalize_base_url


_DEFAULT_USER_AGENT = f"pythea/{__version__}"


class TheaClient:
    """Synchronous client for the Thea Mini Reasoning API."""

    def __init__(
        self,
        *,
        base_url: str,
        apim_subscription_key: Optional[str] = None,
        apim_subscription_key_header: str = "Ocp-Apim-Subscription-Key",
        extra_headers: Optional[Mapping[str, str]] = None,
        timeout_s: float = 60.0,
        verify: Union[bool, str] = True,
        http: Optional[httpx.Client] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.apim_subscription_key = apim_subscription_key
        self.apim_subscription_key_header = apim_subscription_key_header
        self.extra_headers = dict(extra_headers or {})
        self._owns_http = http is None

        self._http = http or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            headers={},
            verify=verify,
            transport=transport,
        )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "TheaClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _build_headers(self, headers: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        base = {
            "Accept": "application/json",
            "User-Agent": _DEFAULT_USER_AGENT,
        }
        out = merge_headers(base, self.extra_headers, headers)
        if self.apim_subscription_key:
            out[self.apim_subscription_key_header] = self.apim_subscription_key
        return out

    def healthz(self, *, headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        """GET /healthz"""
        try:
            r = self._http.get("/healthz", headers=self._build_headers(headers))
        except httpx.TimeoutException as e:
            raise TheaTimeoutError(str(e)) from e
        except httpx.HTTPError as e:
            raise TheaResponseError(str(e)) from e

        if r.status_code >= 400:
            raise TheaHTTPError(
                status_code=r.status_code,
                message="healthz failed",
                response_text=r.text,
                request_id=r.headers.get("X-Request-ID"),
            )

        try:
            return r.json()
        except Exception as e:
            raise TheaResponseError(f"healthz response was not JSON: {e}") from e

    def unified_answer(
        self,
        *,
        question: str,
        evidence: Optional[str] = None,
        backend: Optional[str] = None,
        interpretability: Optional[bool] = None,
        prompt_rewrite: Optional[bool] = None,
        creds: Optional[BackendCredentials] = None,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> UnifiedAnswerResponse:
        """POST /api/unified-answer.

        Everything except `question` is optional; extra keyword args are passed
        through to the JSON request body.
        """

        payload: UnifiedAnswerRequest = {"question": question}
        if evidence is not None:
            payload["evidence"] = evidence
        if backend is not None:
            payload["backend"] = backend
        if interpretability is not None:
            payload["interpretability"] = bool(interpretability)
        if prompt_rewrite is not None:
            payload["prompt_rewrite"] = bool(prompt_rewrite)
        if creds is not None:
            payload["creds"] = creds

        for k, v in kwargs.items():
            if v is None:
                continue
            payload[k] = v  # type: ignore[index]

        try:
            r = self._http.post(
                "/api/unified-answer",
                json=payload,
                headers=self._build_headers(headers),
            )
        except httpx.TimeoutException as e:
            raise TheaTimeoutError(str(e)) from e
        except httpx.HTTPError as e:
            raise TheaResponseError(str(e)) from e

        if r.status_code >= 400:
            msg = None
            try:
                data = r.json()
                if isinstance(data, dict) and "detail" in data:
                    msg = str(data["detail"])
                else:
                    msg = json.dumps(data)
            except Exception:
                msg = r.text.strip() or "request failed"

            raise TheaHTTPError(
                status_code=r.status_code,
                message=msg or "request failed",
                response_text=r.text,
                request_id=r.headers.get("X-Request-ID"),
            )

        try:
            data = r.json()
        except Exception as e:
            raise TheaResponseError(f"unified_answer response was not JSON: {e}") from e

        if not isinstance(data, dict):
            raise TheaResponseError("unified_answer response JSON was not an object")
        return data  # type: ignore[return-value]


class AsyncTheaClient:
    """Async client for the Thea Mini Reasoning API."""

    def __init__(
        self,
        *,
        base_url: str,
        apim_subscription_key: Optional[str] = None,
        apim_subscription_key_header: str = "Ocp-Apim-Subscription-Key",
        extra_headers: Optional[Mapping[str, str]] = None,
        timeout_s: float = 60.0,
        verify: Union[bool, str] = True,
        http: Optional[httpx.AsyncClient] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.apim_subscription_key = apim_subscription_key
        self.apim_subscription_key_header = apim_subscription_key_header
        self.extra_headers = dict(extra_headers or {})
        self._owns_http = http is None

        self._http = http or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            headers={},
            verify=verify,
            transport=transport,
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncTheaClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def _build_headers(self, headers: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        base = {
            "Accept": "application/json",
            "User-Agent": _DEFAULT_USER_AGENT,
        }
        out = merge_headers(base, self.extra_headers, headers)
        if self.apim_subscription_key:
            out[self.apim_subscription_key_header] = self.apim_subscription_key
        return out

    async def healthz(self, *, headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        try:
            r = await self._http.get("/healthz", headers=self._build_headers(headers))
        except httpx.TimeoutException as e:
            raise TheaTimeoutError(str(e)) from e
        except httpx.HTTPError as e:
            raise TheaResponseError(str(e)) from e

        if r.status_code >= 400:
            raise TheaHTTPError(
                status_code=r.status_code,
                message="healthz failed",
                response_text=r.text,
                request_id=r.headers.get("X-Request-ID"),
            )

        try:
            return r.json()
        except Exception as e:
            raise TheaResponseError(f"healthz response was not JSON: {e}") from e

    async def unified_answer(
        self,
        *,
        question: str,
        evidence: Optional[str] = None,
        backend: Optional[str] = None,
        interpretability: Optional[bool] = None,
        prompt_rewrite: Optional[bool] = None,
        creds: Optional[BackendCredentials] = None,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> UnifiedAnswerResponse:
        payload: UnifiedAnswerRequest = {"question": question}
        if evidence is not None:
            payload["evidence"] = evidence
        if backend is not None:
            payload["backend"] = backend
        if interpretability is not None:
            payload["interpretability"] = bool(interpretability)
        if prompt_rewrite is not None:
            payload["prompt_rewrite"] = bool(prompt_rewrite)
        if creds is not None:
            payload["creds"] = creds
        for k, v in kwargs.items():
            if v is None:
                continue
            payload[k] = v  # type: ignore[index]

        try:
            r = await self._http.post(
                "/api/unified-answer",
                json=payload,
                headers=self._build_headers(headers),
            )
        except httpx.TimeoutException as e:
            raise TheaTimeoutError(str(e)) from e
        except httpx.HTTPError as e:
            raise TheaResponseError(str(e)) from e

        if r.status_code >= 400:
            msg = None
            try:
                data = r.json()
                if isinstance(data, dict) and "detail" in data:
                    msg = str(data["detail"])
                else:
                    msg = json.dumps(data)
            except Exception:
                msg = r.text.strip() or "request failed"

            raise TheaHTTPError(
                status_code=r.status_code,
                message=msg or "request failed",
                response_text=r.text,
                request_id=r.headers.get("X-Request-ID"),
            )

        try:
            data = r.json()
        except Exception as e:
            raise TheaResponseError(f"unified_answer response was not JSON: {e}") from e

        if not isinstance(data, dict):
            raise TheaResponseError("unified_answer response JSON was not an object")
        return data  # type: ignore[return-value]
