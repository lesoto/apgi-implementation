import pytest

import numpy as np

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

    # Noise case (seed it)
    np.random.seed(42)
    res_noise = integrate_euler_maruyama(1.0, 0.0, 1.0, 0.0, 1.0)
    # With seed 42, first normal is ~0.4967
    # 1.0 + 0.0*1.0 + 1.0*sqrt(1.0)*0.4967 = 1.4967
    assert pytest.approx(res_noise) == 1.0 + 0.4967141530112327

    with pytest.raises(ValueError, match="dt must be > 0"):
        integrate_euler_maruyama(1.0, 0.5, 0.0, 0.0, 0.0)
