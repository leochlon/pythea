import os
import pytest

from pythea import TheaClient


pytestmark = pytest.mark.e2e


def test_e2e_healthz():
    if os.getenv("THEA_E2E", "").strip() not in {"1", "true", "yes"}:
        pytest.skip("Set THEA_E2E=1 to enable end-to-end tests")

    base_url = os.getenv("THEA_BASE_URL")
    if not base_url:
        pytest.skip("THEA_BASE_URL not set")

    client = TheaClient(
        base_url=base_url,
        apim_subscription_key=os.getenv("THEA_APIM_SUBSCRIPTION_KEY"),
    )
    data = client.healthz()
    assert isinstance(data, dict)
    assert data.get("ok") is True
    client.close()
