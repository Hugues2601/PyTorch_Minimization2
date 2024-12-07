from HestonModels.VanillaFormeAnalytique import *
from HestonModels.old_HestonModel import *
from config import CONFIG
import torch
import numpy as np
import matplotlib.pyplot as plt

phi = torch.tensor(1.0, dtype=torch.complex128, device=CONFIG.device)  # Exemple : une valeur de φ
S0 = torch.tensor(207.0, device=CONFIG.device)
K = torch.tensor([200.0], device=CONFIG.device)  # Exemple : un prix d'exercice
T = torch.tensor([2.0], device=CONFIG.device)
r = torch.tensor(0.0406, device=CONFIG.device)
kappa = torch.tensor(1.461940669421291, device=CONFIG.device)
v0 = torch.tensor(0.10078570773428029, device=CONFIG.device)
theta = torch.tensor(0.121802937587822, device=CONFIG.device)
sigma = torch.tensor(0.414209114742383, device=CONFIG.device)
rho = torch.tensor(-0.6707800066031391, device=CONFIG.device)

# Calcul de la fonction caractéristique
value = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
print("Heston torch Value:", value)

#  # Exemple : une valeur de φ
# phi_numpy = phi.cpu().detach().numpy()
# S0_o = 603
# T_o = 2
# r_o = 0.05
# kappa_o = 1.876
# K_o = 400
# v0_o = 0.0154
# theta_o = 0.2359
# sigma_o = 0.1828
# rho_o = -0.92
#
# # Calcul de la fonction caractéristique
# cf_value = old_heston_price(S0_o, K_o, T_o, r_o, kappa_o, v0_o, theta_o, sigma_o, rho_o)
# old_cf = old_heston_cf(phi_numpy, S0_o, T_o, r_o, kappa_o, v0_o, theta_o, sigma_o, rho_o)
# print("Heston old Value:", cf_value, old_cf)

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters
# S0 = torch.tensor(603.0)
# K = torch.tensor([400.0])
# T = torch.tensor([2.0])
# r = torch.tensor(0.05)
# v0 = torch.tensor(0.05)
# theta = torch.tensor(0.13)
# sigma = torch.tensor(0.72)
# rho = torch.tensor(-0.72)
#
#
# # Varying kappa
# kappa_values = np.linspace(0.1, 4.0, 50)
# torch_prices = []
# numpy_prices = []
#
# for kappa in kappa_values:
#     # Torch computation
#     torch_price = heston_price(S0, K, T, r, torch.tensor(kappa), v0, theta, sigma, rho)
#     torch_prices.append(torch_price.item())
#
#     # Numpy computation
#     numpy_price = old_heston_price(S0_o, K_o, T_o, r_o, kappa, v0_o, theta_o, sigma_o, rho_o)
#     numpy_prices.append(numpy_price)
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(kappa_values, torch_prices, label="Heston Price (Torch)", linestyle='-', marker='o')
# plt.plot(kappa_values, numpy_prices, label="Old Heston Price (Numpy)", linestyle='--', marker='x')
# plt.xlabel("Kappa")
# plt.ylabel("Option Price")
# plt.title("Heston Price vs Old Heston Price with Kappa as Abscissa")
# plt.legend()
# plt.grid()
# plt.show()
