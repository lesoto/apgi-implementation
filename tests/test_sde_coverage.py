import numpy as np
import pytest

from core.sde import integrate_euler_maruyama


def test_integrate_euler_maruyama():
    # Deterministic: x=1.0, mu=0.5, sigma=0.0, dt=0.1
    # x_new = 1.0 + 0.5*0.1 = 1.05
    res = integrate_euler_maruyama(1.0, 0.5, 0.0, 0.0, 0.1)
    assert res == 1.05

    # Callable drift/diffusion
    def mu_fn(x, t):
        return 0.5 * x

    def sigma_fn(x, t):
        return 0.0

    res_fn = integrate_euler_maruyama(1.0, mu_fn, sigma_fn, 0.0, 0.1)
    assert res_fn == 1.05
    # Noise case. Assert against the generator's OWN first draw rather than a
    # magic constant copied from NumPy's legacy global stream, so the test
    # states the Euler-Maruyama identity (x + μ·dt + σ·√dt·N) instead of
    # pinning an implementation detail of one particular RNG.
    expected_normal = float(np.random.default_rng(42).normal(0.0, 1.0))
    res_noise = integrate_euler_maruyama(1.0, 0.0, 1.0, 0.0, 1.0, rng=42)
    assert pytest.approx(res_noise) == 1.0 + 0.0 * 1.0 + 1.0 * np.sqrt(1.0) * expected_normal
    with pytest.raises(ValueError, match="dt must be > 0"):
        integrate_euler_maruyama(1.0, 0.5, 0.0, 0.0, 0.0)
