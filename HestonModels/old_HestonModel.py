import numpy as np

def old_heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho):
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


import numpy as np


def old_heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho):
    params = (S0, T, r, kappa, v0, theta, sigma, rho)
    P1 = 0.5
    P2 = 0.5
    umax = 50  # Limite supérieure de l'intégration
    n = 100  # Nombre de points (doit être impair pour Simpson)

    if n % 2 == 0:
        n += 1  # Assurez-vous que n est impair pour utiliser Simpson correctement

    # Grille pour φ
    phi_values = np.linspace(1e-5, umax, n)
    du = (umax - 1e-5) / (n - 1)  # Pas d'intégration

    # Initialisation des valeurs des intégrandes
    integrand_P1_values = []
    integrand_P2_values = []

    for phi in phi_values:
        factor1 = np.exp(-1j * phi * np.log(K))
        denominator = 1j * phi

        # Calcul de P1
        cf1 = old_heston_cf(phi - 1j, *params) / old_heston_cf(-1j, *params)
        temp1 = factor1 * cf1 / denominator
        integrand_P1_values.append(1 / np.pi * np.real(temp1))

        # Calcul de P2
        cf2 = old_heston_cf(phi, *params)
        temp2 = factor1 * cf2 / denominator
        integrand_P2_values.append(1 / np.pi * np.real(temp2))

    # Convertir en arrays NumPy
    integrand_P1_values = np.array(integrand_P1_values)
    integrand_P2_values = np.array(integrand_P2_values)

    # Règle de Simpson
    weights = np.ones(n)
    weights[1:-1:2] = 4  # Poids pour les points impairs
    weights[2:-2:2] = 2  # Poids pour les points pairs
    weights *= du / 3  # Facteur de Simpson

    integral_P1 = np.sum(weights * integrand_P1_values)
    integral_P2 = np.sum(weights * integrand_P2_values)

    # Calcul des probabilités et du prix
    P1 += integral_P1
    P2 += integral_P2
    price = S0 * P1 - np.exp(-r * T) * K * P2
    return price
