# rge_service/sm_beta.py
import numpy as np

def beta_g_sm_1l(g1, g2, g3):
    # 16π^2 β_gi = b_i g_i^3
    b1, b2, b3 = 41.0/6.0, -19.0/6.0, -7.0
    return b1*g1**3, b2*g2**3, b3*g3**3

def beta_yt_sm_1l(yt, g1, g2, g3):
    # 16π^2 β_y = y [ 9/2 y^2 - (17/12)g1^2 - (9/4)g2^2 - 8 g3^2 ]
    return yt*(4.5*yt**2 - (17.0/12.0)*g1**2 - (9.0/4.0)*g2**2 - 8.0*g3**2)

def beta_lambda_sm_1l(lam, yt, g1, g2, g3):
    # 16π^2 β_λ (SM 1-loop, neglect light Yukawas)
    term_y = 12.0*lam*yt**2 - 12.0*yt**4
    term_g = -lam*(9.0*g2**2 + 3.0*g1**2) + (9.0/8.0)*g2**4 + (3.0/4.0)*g2**2*g1**2 + (3.0/8.0)*g1**4
    term_l = 12.0*lam**2
    return term_l + term_y + term_g

def beta_xi_sm_1l(xi, lam, yt, g1, g2):
    # 16π^2 β_ξ ≈ (ξ - 1/6) (6 y_t^2 - 9/2 g2^2 - 3/2 g1^2 + 12 λ)
    coeff = (6.0*yt**2 - 4.5*g2**2 - 1.5*g1**2 + 12.0*lam)
    return (xi - 1.0/6.0)*coeff
