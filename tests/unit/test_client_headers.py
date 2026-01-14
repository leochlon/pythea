import httpx
import pytest

from pythea import TheaClient


def test_build_headers_includes_subscription_key():
    client = TheaClient(base_url="http://example.com", apim_subscription_key="abc123")
    headers = client._build_headers()
    assert headers["Ocp-Apim-Subscription-Key"] == "abc123"
    assert headers["Accept"] == "application/json"
    assert "User-Agent" in headers
    client.close()


def test_build_headers_allows_override_and_extra_headers():
    client = TheaClient(
        base_url="http://example.com",
        apim_subscription_key="abc123",
        extra_headers={"X-Foo": "bar"},
    )
    headers = client._build_headers({"X-Foo": "baz", "X-Bar": "qux"})
    assert headers["X-Foo"] == "baz"
    assert headers["X-Bar"] == "qux"
    client.close()
