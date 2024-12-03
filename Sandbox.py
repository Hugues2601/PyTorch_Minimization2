from HestonModels.VanillaFormeAnalytique import heston_cf
from HestonModels.old_HestonModel import old_heston_cf
from config import CONFIG
import torch
import numpy as np

# phi = torch.tensor(1.0, dtype=torch.complex128, device=CONFIG.device)  # Exemple : une valeur de φ
# S0 = torch.tensor(100.0, device=CONFIG.device)
# T = torch.tensor(1.0, device=CONFIG.device)
# r = torch.tensor(0.03, device=CONFIG.device)
# kappa = torch.tensor(1.911, device=CONFIG.device)
# v0 = torch.tensor(0.0095, device=CONFIG.device)
# theta = torch.tensor(0.0099, device=CONFIG.device)
# sigma = torch.tensor(0.2893, device=CONFIG.device)
# rho = torch.tensor(-0.5889, device=CONFIG.device)
#
# # Calcul de la fonction caractéristique
# cf_value = heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho)
# print("Heston CF Value:", cf_value)

phi = 1.0  # Exemple : une valeur de φ
S0 = 100.0
T = 1.0
r = 0.03
kappa = 1.911
v0 = 0.0095
theta = 0.0099
sigma = 0.2893
rho = -0.5889

# Calcul de la fonction caractéristique
cf_value = old_heston_cf(phi, S0, T, r, kappa, v0, theta, sigma, rho)
print("Heston CF Value:", cf_value)