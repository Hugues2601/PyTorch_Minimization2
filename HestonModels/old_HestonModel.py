import numpy as np

def heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
    a = -0.5 * phi ** 2 - 0.5j * phi
    b = kappa - rho * sigma * 1j * phi
    g = ((b - np.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2) / (
                (b + np.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2)
    C = kappa * (((b - np.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2) * T - 2 / sigma ** 2 * np.log(
        (1 - g * np.exp(-np.sqrt(b ** 2 - 2 * sigma ** 2 * a) * T)) / (1 - g)))
    D = ((b - np.sqrt(b ** 2 - 2 * sigma ** 2 * a)) / sigma ** 2) * (
                1 - np.exp(-np.sqrt(b ** 2 - 2 * sigma ** 2 * a) * T)) / (
                    1 - g * np.exp(-np.sqrt(b ** 2 - 2 * sigma ** 2 * a) * T))

    cf = np.exp(C * theta + D * v0 + 1j * phi * np.log(S0 * np.exp(r * T)))
    return cf


# In[9]:


def heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho):
    params = (S0, T, r, kappa, v0, theta, sigma, rho)
    P1 = 0.5
    P2 = 0.5
    umax = 50
    n = 100
    du = umax / n
    phi = du / 2
    for i in range(n):
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi
        cf1 = heston_cf(phi - 1j, *params) / heston_cf(-1j, *params)
        temp1 = factor1 * cf1 / denominator
        P1 += 1 / np.pi * np.real(temp1) * du
        cf2 = heston_cf(phi, *params)
        temp2 = factor1 * cf2 / denominator
        P2 += 1 / np.pi * np.real(temp2) * du
        phi += du
    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price