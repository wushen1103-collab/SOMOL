import numpy as np

from scripts.run_exact_no_gauge_ablation import simplex_grid
from scripts.run_method_comparison_baselines import fit_deming, fit_tls, fit_trimmed_ols


def test_deming_and_tls_recover_affine_relation():
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 0.75 + 1.5 * x
    for model in (fit_deming(x, y), fit_tls(x, y)):
        assert np.isclose(model.intercept, 0.75)
        assert np.isclose(model.slope, 1.5)


def test_trimmed_fit_suppresses_single_outlier():
    x = np.arange(10, dtype=float)
    y = 2.0 + 0.5 * x
    y[-1] += 50.0
    model = fit_trimmed_ols(x, y, trim_fraction=0.2)
    assert np.isclose(model.intercept, 2.0)
    assert np.isclose(model.slope, 0.5)


def test_five_component_simplex_grid():
    weights = simplex_grid(components=5, step=0.1)
    assert len(weights) == 1001
    assert all(np.isclose(sum(weight), 1.0) for weight in weights)
