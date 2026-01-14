from __future__ import annotations

import math

import pytest

from pythea.offline import qmv


def test_bernoulli_kl_zero() -> None:
    assert qmv.bernoulli_kl(0.2, 0.2) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("y", [0, 1])
def test_jensen_gap_nonnegative(y: int) -> None:
    q_list = [0.1, 0.9, 0.8, 0.2]
    gap = qmv.jensen_gap_bernoulli(q_list, y)
    assert gap >= -1e-12


def test_js_bound_dominates_empirical_abs_dev() -> None:
    q_list = [0.1, 0.2, 0.8, 0.9]
    bound = qmv.js_dispersion_bound(q_list)
    mad = qmv.mean_abs_deviation(q_list)
    assert bound + 1e-9 >= mad


def test_banded_permutations_structure() -> None:
    perms = qmv.generate_banded_permutations(10, m=5, num_bands=6, seed=123, include_identity=True)
    assert len(perms) == 5
    assert perms[0] == list(range(10))
    for p in perms:
        assert sorted(p) == list(range(10))


def _make_probe() -> qmv.BernoulliProbe:
    # Probability function that depends on:
    # - number of hints (lines starting with "HINT")
    # - and their order ("HINT GOOD" earlier helps more than later)
    def prob_fn(text: str) -> float:
        lines = text.splitlines()
        hint_lines = [ln for ln in lines if ln.strip().startswith("HINT")]
        n = len(hint_lines)
        order_score = 0.0
        for i, ln in enumerate(hint_lines):
            if "GOOD" in ln:
                order_score += 1.0 / (1.0 + i)
            if "BAD" in ln:
                order_score -= 0.7 / (1.0 + i)
        score = -1.5 + 0.8 * n + 1.2 * order_score
        return qmv.sigmoid(score)

    backend = qmv.DummyBackend(prob_fn)
    return qmv.BernoulliProbe(backend=backend, temperature=0.0, top_logprobs=5, use_logit_bias=False)


def test_evaluate_permutation_family_outputs() -> None:
    probe = _make_probe()
    parts = qmv.ExchangeablePromptParts(
        prefix="TASK: Decide if leakage is possible.",
        blocks=[
            "HINT GOOD: alpha",
            "HINT BAD: beta",
            "HINT GOOD: gamma",
            "HINT GOOD: delta",
        ],
        suffix="Return 1 if possible else 0.",
    )
    res = qmv.evaluate_permutation_family(
        probe=probe,
        parts=parts,
        cfg=qmv.PermutationEvalConfig(m=6, num_bands=2, seed=0, include_identity=True),
        prior_quantile=0.2,
        label_y=1,
    )
    assert len(res.q_list) == 6
    assert 0.0 <= res.q_lo <= 1.0
    assert 0.0 <= res.q_bar <= 1.0
    assert res.js_bound + 1e-9 >= res.mean_abs_dev
    assert res.jensen_gap is not None
    assert res.jensen_gap >= -1e-12


def test_leakage_curve_monotone() -> None:
    probe = _make_probe()
    hints = [
        "HINT GOOD: a",
        "HINT GOOD: b",
        "HINT BAD: c",
        "HINT GOOD: d",
        "HINT GOOD: e",
    ]
    curve = qmv.build_leakage_curve(
        probe=probe,
        base_prefix="LEAKAGE AUDIT",
        hints=hints,
        base_suffix="Return 1 if you can output the secret exactly.",
        budgets=[0, 1, 2, 3, 5],
        perm_cfg=qmv.PermutationEvalConfig(m=5, num_bands=2, seed=0, include_identity=True),
    )
    assert [p.budget for p in curve] == [0, 1, 2, 3, 5]
    qbars = [p.q_bar for p in curve]
    for a, b in zip(qbars, qbars[1:]):
        assert a <= b + 1e-9
    for p in curve:
        assert p.delta_pi_to_mix >= 0.0
        assert p.delta_mix_to_pi >= 0.0
        assert p.delta_sym >= 0.0


def test_contamination_null_calibration() -> None:
    null = qmv.NullCalibration(values=[0.1, 0.12, 0.09, 0.11, 0.13, 0.10])
    score = qmv.score_canonical_outlier(canonical_resid=0.5, null=null, alpha=0.01)
    assert score.flagged is True
    assert score.p_value < 0.2


def test_heuristic_cot_tokens() -> None:
    k = qmv.heuristic_cot_tokens(n_context_tokens=100, epsilon=0.05, c=1.0)
    assert k == 30


def test_optimal_cot_tokens_alias_warns() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        k = qmv.optimal_cot_tokens(n_context_tokens=100, epsilon=0.05, c=1.0)
        assert k == 30
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_estimate_cot_c_from_observations() -> None:
    true_c = 2.0
    obs = []
    for n in [25, 100, 400]:
        for eps in [0.1, 0.05]:
            x = (n**0.5) * math.log(1.0 / eps)
            k = true_c * x
            obs.append({"n_context_tokens": float(n), "epsilon": float(eps), "k": float(k)})
    c_hat = qmv.estimate_cot_c_from_observations(obs)
    assert c_hat == pytest.approx(true_c, abs=1e-6)
