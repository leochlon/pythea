from __future__ import annotations

from typing import Dict, Mapping, Optional


def normalize_base_url(base_url: str) -> str:
    """Normalize a base URL for httpx's base_url usage."""
    if not base_url:
        raise ValueError("base_url is required")
    return base_url.rstrip("/")


def merge_headers(*parts: Optional[Mapping[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in parts:
        if not p:
            continue
        out.update({k: v for k, v in p.items() if v is not None})
    return out
