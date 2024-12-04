from HestonModels.VanillaFormeAnalytique import *
from HestonModels.old_HestonModel import *
from config import CONFIG
import torch
import numpy as np

phi = torch.tensor(1.0, dtype=torch.complex128, device=CONFIG.device)  # Exemple : une valeur de φ
S0 = torch.tensor(200.0, device=CONFIG.device)
K = torch.tensor([145.0], device=CONFIG.device)  # Exemple : un prix d'exercice
T = torch.tensor([2.0], device=CONFIG.device)
r = torch.tensor(0.03, device=CONFIG.device)
kappa = torch.tensor(4.5, device=CONFIG.device)
v0 = torch.tensor(0.01, device=CONFIG.device)
theta = torch.tensor(0.01, device=CONFIG.device)
sigma = torch.tensor(0.29, device=CONFIG.device)
rho = torch.tensor(-0.59, device=CONFIG.device)

# Calcul de la fonction caractéristique
value = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
print("Heston torch Value:", value)

 # Exemple : une valeur de φ
S0_o = 200.0
T_o = 2.0
r_o = 0.03
kappa_o = 4.5
K_o = 145
v0_o = 0.01
theta_o = 0.01
sigma_o = 0.29
rho_o = -0.59

# Calcul de la fonction caractéristique
cf_value = old_heston_price(S0_o, K_o, T_o, r_o, kappa_o, v0_o, theta_o, sigma_o, rho_o)
print("Heston old Value:", cf_value)