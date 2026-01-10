""" 
Simple AOAI pool client for Python (optimized)

This is a drop-in replacement for aoai_pool_python.py with the same public API:
  - AoaiPool.from_file(path)
  - AoaiPool.chat(messages, **options)

Optimizations / hardening:
  - Fixes double-counting of requests in the RPM tracker (which could cause
    premature local throttling and unnecessary sleeping).
  - Uses a thread-local requests.Session for connection reuse (faster, lower CPU).
  - Uses deque-based sliding windows for RPM/TPM counters (lower overhead under load).
  - Updates BOTH TPM and RPM limits from Azure headers when provided.

Behavior is otherwise intentionally conservative and compatible.
"""

from __future__ import annotations

import json
import os
import random
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests
from email.utils import parsedate_to_datetime


# =========================
# Helpers
# =========================

def _now_ms() -> float:
    return time.time() * 1000.0


def _debug_enabled() -> bool:
    return os.getenv("AOAI_POOL_DEBUG", "").strip() not in ("", "0", "false", "False")


def _dprint(msg: str) -> None:
    if _debug_enabled():
        print(f"[AOAI_POOL] {msg}")


def _parse_retry_after_ms(headers: Dict[str, str]) -> Optional[int]:
    """Parse Azure/OpenAI retry hints, preferring millisecond precision."""
    # 1) Millisecond headers first
    for k in ("retry-after-ms", "x-ms-retry-after-ms"):
        v = headers.get(k)
        if v is None:
            continue
        v = str(v).strip()
        if v.isdigit():
            try:
                ms = int(v)
                if ms >= 0:
                    return ms
            except Exception:
                pass

    # 2) Standard Retry-After (seconds or HTTP-date)
    ra = headers.get("retry-after")
    if ra is None:
        return None
    ra = str(ra).strip()
    # If it's a number, it's seconds
    if ra.replace(".", "", 1).isdigit():
        try:
            seconds = float(ra)
            if seconds >= 0:
                return int(seconds * 1000.0)
        except Exception:
            pass
    # Otherwise try HTTP-date
    try:
        dt = parsedate_to_datetime(ra)
        delta = (dt.timestamp() * 1000.0) - _now_ms()
        if delta > 0:
            return int(delta)
    except Exception:
        pass
    return None


def _exp_backoff_ms(base_ms: int, attempt: int, cap_ms: int) -> int:
    """Exponential backoff with full jitter (attempt is 0-based)."""
    max_sleep = min(cap_ms, max(base_ms, base_ms * (2 ** attempt)))
    return random.randint(max(1, base_ms), max(1, max_sleep))


def _sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


# =========================
# Backend model
# =========================


@dataclass
class Backend:
    id: str
    endpoint: str
    deployment: str
    apiKey: str
    apiVersion: str = "2024-06-01"
    weight: int = 1
    inFlight: int = 0
    healthy: bool = True
    cooldownUntil: float = 0.0  # epoch ms
    recoverAt: float = 0.0      # epoch ms

    # Rate limit tracking (best-effort local limiter; per-process)
    requestTimes: Deque[float] = None
    tokenUsage: Deque[Tuple[float, int]] = None
    rpmLimit: int = 180
    tpmLimit: int = 60000

    def __post_init__(self) -> None:
        if self.requestTimes is None:
            self.requestTimes = deque()
        if self.tokenUsage is None:
            self.tokenUsage = deque()

    def _prune(self, now_ms: float) -> None:
        cutoff = now_ms - 60_000
        while self.requestTimes and self.requestTimes[0] <= cutoff:
            self.requestTimes.popleft()
        while self.tokenUsage and self.tokenUsage[0][0] <= cutoff:
            self.tokenUsage.popleft()

    def record_request(self, now_ms: float, tokens: int = 0) -> None:
        """Record a request timestamp and (optionally) token usage.

        IMPORTANT: This should be called at most once per HTTP request.
        If you want to record tokens after the response, use record_tokens().
        """
        self.requestTimes.append(now_ms)
        if tokens > 0:
            self.tokenUsage.append((now_ms, int(tokens)))
        self._prune(now_ms)

    def record_tokens(self, now_ms: float, tokens: int) -> None:
        """Record token usage WITHOUT adding an extra request timestamp.

        This prevents double-counting the request in the RPM window.
        """
        if tokens and tokens > 0:
            self.tokenUsage.append((now_ms, int(tokens)))
        self._prune(now_ms)

    def requests_in_last_minute(self, now_ms: float) -> int:
        self._prune(now_ms)
        return len(self.requestTimes)

    def tokens_in_last_minute(self, now_ms: float) -> int:
        self._prune(now_ms)
        return sum(tok for _, tok in self.tokenUsage)

    def is_rate_limited(self, now_ms: float) -> bool:
        self._prune(now_ms)
        if self.rpmLimit > 0 and len(self.requestTimes) >= self.rpmLimit:
            return True
        if self.tpmLimit > 0 and self.tokens_in_last_minute(now_ms) >= self.tpmLimit:
            return True
        return False

    def is_available(self, now_ms: float) -> bool:
        if (not self.healthy) and now_ms >= self.recoverAt:
            self.healthy = True
        if now_ms < self.cooldownUntil:
            return False
        if self.is_rate_limited(now_ms):
            return False
        return self.healthy

    def next_available_at(self) -> float:
        if self.healthy:
            return max(self.cooldownUntil, 0.0)
        return max(self.cooldownUntil, self.recoverAt)


# =========================
# Pool
# =========================


class AoaiPool:
    def __init__(self, backends: List[Dict[str, Any]]):
        if not backends:
            raise ValueError("AoaiPool requires at least one backend")

        self.backends: List[Backend] = [Backend(**b) for b in backends]
        for b in self.backends:
            b.weight = max(1, int(b.weight))

        self.totalWeight = sum(b.weight for b in self.backends)
        self._rrCursor = 0
        self._kv_cache: Dict[str, Dict[str, Any]] = {}

        # Selection & inflight accounting
        self._lock = threading.RLock()

        # Thread-local HTTP sessions for connection reuse
        self._tls = threading.local()

        self._default_timeout_s = 60.0

    @classmethod
    def from_file(cls, path: str) -> "AoaiPool":
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        return cls(arr)

    # ----- HTTP session -----

    def _get_session(self) -> requests.Session:
        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = requests.Session()
            # A small but useful pool size; caller concurrency is usually modest.
            # If you expect many threads, set AOAI_POOL_CONN_POOL_MAXSIZE.
            pool_max = int(os.getenv("AOAI_POOL_CONN_POOL_MAXSIZE", "32"))
            pool_conns = int(os.getenv("AOAI_POOL_CONN_POOL_CONNECTIONS", "32"))
            adapter = requests.adapters.HTTPAdapter(pool_connections=pool_conns, pool_maxsize=pool_max, max_retries=0)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._tls.session = sess
        return sess

    # ----- Weighted RR -----

    def _next_weighted(self) -> Backend:
        with self._lock:
            target = self._rrCursor % self.totalWeight
            self._rrCursor = (self._rrCursor + 1) & 0x7FFFFFFF
        acc = 0
        for b in self.backends:
            acc += b.weight
            if target < acc:
                return b
        return self.backends[0]

    # ----- Selection with cooldown awareness -----

    def _select_backend(self, *, jitter_max_ms: int, max_sleep_ms: int) -> Backend:
        """Select least-loaded available backend; sleep until earliest availability if none."""
        jitter_max_ms = max(0, int(jitter_max_ms))
        while True:
            now = _now_ms()
            best: Optional[Backend] = None
            best_score = float("inf")

            with self._lock:
                n = len(self.backends)
                for _ in range(n):
                    cand = self._next_weighted()
                    if not cand.is_available(now):
                        continue
                    score = cand.inFlight / max(1, cand.weight)
                    if score < best_score:
                        best = cand
                        best_score = score

                if best is not None:
                    best.inFlight += 1
                    # Record a single request timestamp here (counts toward RPM).
                    best.record_request(now)
                    return best

            # None available: wait for earliest next_available_at
            next_times = [max(1.0, b.next_available_at() - now) for b in self.backends]
            sleep_ms = int(min(next_times))
            jitter = random.randint(0, jitter_max_ms) if jitter_max_ms > 0 else 0
            total_sleep = min(max_sleep_ms, sleep_ms + jitter)
            _dprint(f"All backends cooling/recovering; sleeping {total_sleep} ms")
            _sleep_ms(total_sleep)

    # ----- Key Vault resolution (unchanged) -----

    def _resolve_api_key(self, b: Backend) -> str:
        key = b.apiKey or ""
        if not key.lower().startswith("kv://"):
            return key
        # kv://<vault>/<secret>[#version]
        try:
            _, rest = key.split("://", 1)
            vault, name_version = rest.split("/", 1)
            if "#" in name_version:
                name, version = name_version.split("#", 1)
            else:
                name, version = name_version, None
        except Exception:
            raise ValueError(f"Invalid Key Vault reference: {key}")

        kv_uri = f"kv://{vault}/{name}{('#' + version) if version else ''}"
        now = _now_ms()
        cached = self._kv_cache.get(kv_uri)
        if cached and cached.get("expiresAt", 0) > now:
            return cached["value"]

        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            from azure.keyvault.secrets import SecretClient  # type: ignore
        except Exception as e:
            raise ImportError("Install azure-identity and azure-keyvault-secrets to use Key Vault refs.") from e

        url = f"https://{vault}.vault.azure.net"
        client = SecretClient(url, DefaultAzureCredential())
        resp = client.get_secret(name, version=version)
        value = resp.value
        if not value:
            raise RuntimeError(f"Secret {name} in {vault} is empty")
        ttl = int(os.getenv("KV_CACHE_TTL_MS", "600000"))
        self._kv_cache[kv_uri] = {"value": value, "expiresAt": now + ttl}
        return value

    # ----- Core chat -----

    def chat(self, messages: List[Dict[str, Any]], **options: Any) -> Dict[str, Any]:
        """Non-streaming chat.completions against Azure OpenAI using the pool."""
        body: Dict[str, Any] = {
            "messages": messages,
            "temperature": options.get("temperature", 0.2),
            "max_tokens": options.get("max_tokens"),
            "stream": False,
        }
        for key in [
            "tools", "tool_choice", "n", "response_format",
            "top_p", "frequency_penalty", "presence_penalty",
            "stop", "seed",
        ]:
            if key in options:
                body[key] = options[key]
        extra = options.get("extra")
        if isinstance(extra, dict):
            body.update(extra)

        # Controls
        max_attempts = int(options.get("max_attempts", len(self.backends)))
        base_delay_ms = int(options.get("base_delay_ms", 5000))  # used when Retry-After absent
        backoff_cap_ms = int(options.get("backoff_cap_ms", 120_000))
        timeout_s = float(options.get("timeout_ms", int(self._default_timeout_s * 1000))) / 1000.0
        global_retry_cap_ms = int(options.get("global_retry_cap_ms", 60_000))
        headers_extra: Dict[str, str] = dict(options.get("headers") or {})

        # IMPORTANT: jitter max should be small even if base_delay_ms is large.
        jitter_max_ms = int(options.get("jitter_max_ms", min(250, max(0, base_delay_ms))))

        if max_attempts <= 0:
            max_attempts = 1

        last_err: Optional[Exception] = None
        session = self._get_session()

        for attempt in range(max_attempts):
            b = self._select_backend(jitter_max_ms=jitter_max_ms, max_sleep_ms=backoff_cap_ms)

            url = f"{b.endpoint.rstrip('/')}/openai/deployments/{b.deployment}/chat/completions"
            params = {"api-version": b.apiVersion}
            api_key = self._resolve_api_key(b)
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
                "User-Agent": "aoai-pool/1.2-opt",
            }
            headers.update(headers_extra)

            try:
                _dprint(f"Attempt {attempt+1}/{max_attempts} -> {b.id} (inFlight={b.inFlight}, weight={b.weight})")
                res = session.post(url, headers=headers, params=params, json=body, timeout=timeout_s)

                # Debug / diagnostics
                if _debug_enabled():
                    rem_req = res.headers.get("x-ratelimit-remaining-requests")
                    rem_tok = res.headers.get("x-ratelimit-remaining-tokens")
                    req_id = res.headers.get("x-request-id") or res.headers.get("x-ms-request-id")
                    _dprint(f"{b.id} status={res.status_code} x-rr={rem_req} x-rt={rem_tok} req={req_id}")

                # Update limits from headers when available (helps local limiter match reality)
                try:
                    limit_tokens = res.headers.get("x-ratelimit-limit-tokens")
                    if limit_tokens and str(limit_tokens).strip().isdigit():
                        b.tpmLimit = int(limit_tokens)
                    limit_reqs = res.headers.get("x-ratelimit-limit-requests")
                    if limit_reqs and str(limit_reqs).strip().isdigit():
                        b.rpmLimit = int(limit_reqs)
                except Exception:
                    pass

                # Throttling / transient unavailability
                if res.status_code in (429, 503):
                    retry_ms = _parse_retry_after_ms(dict(res.headers))
                    if retry_ms is None:
                        retry_ms = _exp_backoff_ms(base_delay_ms, attempt, backoff_cap_ms)
                    retry_ms = min(int(retry_ms), global_retry_cap_ms)

                    now = _now_ms()
                    b.cooldownUntil = now + retry_ms
                    last_err = RuntimeError(f"Backend {b.id} throttled: {res.status_code}")
                    _dprint(f"{b.id} throttled {res.status_code}; cooldown {retry_ms} ms")
                    continue

                # Other server errors mark unhealthy briefly
                if res.status_code >= 500:
                    b.healthy = False
                    b.recoverAt = _now_ms() + 30_000
                    last_err = RuntimeError(f"HTTP {res.status_code} from {b.id}: {res.text[:200]}")
                    _dprint(f"{b.id} marked unhealthy (5xx)")
                    continue

                # Raise on other client errors (4xx)
                res.raise_for_status()

                # Extract token usage; record tokens WITHOUT incrementing RPM again.
                data = res.json()
                try:
                    usage = data.get("usage", {}) if isinstance(data, dict) else {}
                    tokens_used = int(usage.get("total_tokens", 0) or 0)
                    if tokens_used > 0:
                        with self._lock:
                            b.record_tokens(_now_ms(), tokens_used)
                        _dprint(
                            f"{b.id} used {tokens_used} tokens, TPM: {b.tokens_in_last_minute(_now_ms())}/{b.tpmLimit}"
                        )
                except Exception:
                    pass

                return data

            except requests.RequestException as e:
                last_err = e
                # Network-level errors: tiny backoff, try another backend.
                sleep_ms = min(100, _exp_backoff_ms(max(1, base_delay_ms // 10), attempt, backoff_cap_ms))
                _dprint(f"Network error on {b.id}: {e}; brief sleep {sleep_ms} ms")
                _sleep_ms(int(sleep_ms))

            finally:
                with self._lock:
                    b.inFlight = max(0, b.inFlight - 1)

        if last_err is not None:
            raise last_err
        raise RuntimeError("All backends failed (no exception captured)")
