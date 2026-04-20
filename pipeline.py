from core.preprocessing import compute_prediction_error, z_score, RunningStats
from core.precision import (
    compute_precision,
    compute_effective_interoceptive_precision
)
from core.signal import compute_apgi_signal
from core.ignition import detect_ignition_event, compute_ignition_probability
from hierarchy.multiscale import apply_reset_rule


class APGIPipeline:
    def __init__(self, config):
        self.config = config
        self.stats_e = RunningStats(window_size=config['window_size'])
        self.stats_i = RunningStats(window_size=config['window_size'])
        self.theta = config['theta_0']
        self.S = 0.0

    def step(self, x_e, x_hat_e, x_i, x_hat_i, somatic_marker):
        # 1. Prediction Error
        eps_e = compute_prediction_error(x_e, x_hat_e)
        eps_i = compute_prediction_error(x_i, x_hat_i)

        self.stats_e.update(eps_e)
        self.stats_i.update(eps_i)

        # 2. Standardization
        z_e = z_score(eps_e, self.stats_e)
        z_i = z_score(eps_i, self.stats_i)

        # 3. Precision
        pi_e = compute_precision(self.stats_e.std())
        pi_i_base = compute_precision(self.stats_i.std())

        # 4. Somatic Bias
        pi_i_eff = compute_effective_interoceptive_precision(
            pi_i_base, self.config['beta'], somatic_marker
        )

        # 5. Signal
        self.S = compute_apgi_signal(z_e, z_i, pi_e, pi_i_eff)

        # 6. Ignition Probability
        B_t = compute_ignition_probability(
            self.S, self.theta, self.config['alpha']
        )

        # 7. Ignition Detection
        ignition = detect_ignition_event(self.S, self.theta)

        # 8. Reset Rule
        if ignition:
            self.S, self.theta = apply_reset_rule(
                self.S, self.theta,
                rho=self.config['rho'],
                delta=self.config['delta']
            )

        return {
            'S': self.S,
            'theta': self.theta,
            'ignition': ignition,
            'B_t': B_t
        }
