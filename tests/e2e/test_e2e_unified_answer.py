import os
import pytest

from pythea import TheaClient
from pythea.errors import TheaHTTPError


pytestmark = pytest.mark.e2e


def _build_creds(backend: str):
    backend = (backend or "").strip().lower()

    # The server can be configured so some backends require per-request creds.
    # We only include creds if the relevant env vars are present; otherwise we skip.
    if backend == "openai":
        key = os.getenv("THEA_OPENAI_API_KEY")
        if not key:
            return None
        return {"openai_api_key": key}

    if backend == "azure":
        key = os.getenv("THEA_AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("THEA_AZURE_OPENAI_ENDPOINT")
        if not (key and endpoint):
            return None
        creds = {"azure_api_key": key, "azure_endpoint": endpoint}
        ver = os.getenv("THEA_AZURE_OPENAI_API_VERSION")
        if ver:
            creds["azure_api_version"] = ver
        return creds

    # aoai-pool/openrouter/etc may not require creds depending on your deployment.
    return None


def test_e2e_unified_answer_minimal():
    if os.getenv("THEA_E2E", "").strip() not in {"1", "true", "yes"}:
        pytest.skip("Set THEA_E2E=1 to enable end-to-end tests")

    base_url = os.getenv("THEA_BASE_URL")
    if not base_url:
        pytest.skip("THEA_BASE_URL not set")

    backend = os.getenv("THEA_BACKEND", "aoai-pool")
    creds = _build_creds(backend)

    # If this backend needs creds but we don't have them, skip cleanly.
    if backend in {"openai", "azure"} and creds is None:
        pytest.skip(f"Backend {backend} requires creds; set THEA_* credentials env vars")

    client = TheaClient(
        base_url=base_url,
        apim_subscription_key=os.getenv("THEA_APIM_SUBSCRIPTION_KEY"),
    )

    try:
        resp = client.unified_answer(
            question="What is 2+2?",
            backend=backend,
            creds=creds,
            # interpretability=True triggers the richer trace in many deployments (and is
            # what you typically want to validate before deploying).
            interpretability=True,
            prompt_rewrite=False,
        )
    except TheaHTTPError as e:
        # For dev environments with strict plan/auth rules, surface the message,
        # but still fail (this is intended to be an actual E2E).
        raise

    assert isinstance(resp, dict)
    # Accept both minimal and full response shapes
    if "status" in resp:
        assert resp["status"] in {"answered", "abstained"}
        if resp["status"] == "answered":
            assert resp.get("answer") is not None
    else:
        assert "decision" in resp

    client.close()
