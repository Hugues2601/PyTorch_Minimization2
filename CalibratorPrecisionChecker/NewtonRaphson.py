import torch
from config import CONFIG
from HestonModels.VanillaFormeAnalytique import heston_price
from DataProcessing import data_processing
from results import Results
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math


file, market_prices, T_list, IV, K_list = data_processing(CONFIG.df)
import torch

# Paramètres calibrés
kappa_val = 1.461940669421291
v0_val    = 0.10078570773428029
theta_val = 0.121802937587822
sigma_val = 0.2814209114742383
rho_val   = -0.6707800066031391
S0_val    = 207.0
r_val     = 0.0406

# Conversion en tenseurs
K_t = torch.tensor(K_list, dtype=torch.float64, device=CONFIG.device)
T_t = torch.tensor(T_list, dtype=torch.float64, device=CONFIG.device)

# Création des paramètres en tenseurs avec requires_grad=True
S0 = torch.tensor(S0_val, dtype=torch.float64, device=CONFIG.device, requires_grad=True)
r  = torch.tensor(r_val,  dtype=torch.float64, device=CONFIG.device, requires_grad=True)

kappa = torch.tensor(kappa_val, dtype=torch.float64, device=CONFIG.device, requires_grad=True)
v0    = torch.tensor(v0_val,    dtype=torch.float64, device=CONFIG.device, requires_grad=True)
theta = torch.tensor(theta_val, dtype=torch.float64, device=CONFIG.device, requires_grad=True)
sigma = torch.tensor(sigma_val, dtype=torch.float64, device=CONFIG.device, requires_grad=True)
rho   = torch.tensor(rho_val,   dtype=torch.float64, device=CONFIG.device, requires_grad=True)

S0.grad = None
r.grad = None
sigma.grad = None
# Calcul du prix selon Heston
price = heston_price(S0, K_t, T_t, r, kappa, v0, theta, sigma, rho)

# price est un tenseur 1D, par exemple de taille len(K_list).
# Pour obtenir un gradient, on a besoin d'un scalaire. Prenons la moyenne.
loss = price.mean()

# Calcul des gradients
loss.backward()

# Delta par rapport à S0 est maintenant dans S0.grad
vega = sigma.grad.item()# item() pour récupérer la valeur scalaire

print("Delta (dPrix/dS0) =", vega)
