def compute_apgi_signal(
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i_eff: float
) -> float:
    return pi_e * abs(z_e) + pi_i_eff * abs(z_i)
