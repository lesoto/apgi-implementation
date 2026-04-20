def update_threshold_discrete(theta, metabolic_cost, information_value, eta=0.1):
    return theta + eta * (metabolic_cost - information_value)


def update_threshold_ode(theta, theta_0, dS_dt, B_prev, gamma, delta, lam):
    return gamma * (theta_0 - theta) + delta * B_prev - lam * abs(dS_dt)
