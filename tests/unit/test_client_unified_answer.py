import json
import httpx
import pytest

from pythea import TheaClient
from pythea.errors import TheaHTTPError


def test_unified_answer_posts_expected_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/unified-answer"
        body = json.loads(request.content.decode("utf-8"))
        seen["body"] = body
        # minimal fake server response
        return httpx.Response(200, json={"decision": "answer", "passed": True, "picked": "4", "backend": body.get("backend", "openai")})

    transport = httpx.MockTransport(handler)
    client = TheaClient(base_url="http://testserver", transport=transport)

    resp = client.unified_answer(
        question="What is 2+2?",
        backend="aoai-pool",
        interpretability=True,
        m=3,
    )

    assert resp["decision"] == "answer"
    assert seen["body"]["question"] == "What is 2+2?"
    assert seen["body"]["backend"] == "aoai-pool"
    assert seen["body"]["interpretability"] is True
    assert seen["body"]["m"] == 3

    client.close()


def test_unified_answer_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad request"})

    transport = httpx.MockTransport(handler)
    client = TheaClient(base_url="http://testserver", transport=transport)

    with pytest.raises(TheaHTTPError) as e:
        client.unified_answer(question="x")
    assert e.value.status_code == 400
    assert "bad request" in str(e.value)

    client.close()
