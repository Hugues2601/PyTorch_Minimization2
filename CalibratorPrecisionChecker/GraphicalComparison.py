from HestonModels.old_HestonModel import old_heston_price
from HestonModels.VanillaFormeAnalytique import heston_price
from config import CONFIG
from results import Results
from matplotlib import pyplot as plt
import torch
import numpy as np


# Définir l'appareil (GPU si disponible, sinon CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paramètres de calibration
calibration_results = {
    'kappa': 0.3390940971835357,
    'v0': 0.054291934542661895,
    'theta': 0.12895449836310516,
    'sigma': 0.7180440005560889,
    'rho': -0.7239568444613573
}

# Conversion des paramètres en tenseurs PyTorch sur l'appareil approprié
kappa = torch.tensor(calibration_results['kappa'], dtype=torch.float64, device=device)
v0 = torch.tensor(calibration_results['v0'],  dtype=torch.float64, device=device)
theta = torch.tensor(calibration_results['theta'],dtype=torch.float64, device=device)
sigma = torch.tensor(calibration_results['sigma'],  dtype=torch.float64, device=device)
rho = torch.tensor(calibration_results['rho'], dtype=torch.float64, device=device)

# Autres constantes
S0 = torch.tensor(603, dtype=torch.float64, device=device)
r = torch.tensor(0.0406, dtype=torch.float64, device=device)

# Définir la valeur de K
K_single = 600

# Calculer le prix avec heston_price
heston_price_single = heston_price(
    S0=S0,
    K=torch.tensor([K_single], dtype=torch.float64, device=device),
    T=torch.tensor([1.0], dtype=torch.float64, device=device),
    r=r,
    kappa=kappa,
    v0=v0,
    theta=theta,
    sigma=sigma,
    rho=rho
)

# Calculer le prix avec old_heston_price
old_heston_price_single = old_heston_price(
    S0=603, K=600, T=1.0, r=0.0406,
    kappa=calibration_results['kappa'], v0=calibration_results['v0'],
    theta=calibration_results['theta'], sigma=calibration_results['sigma'],
    rho=calibration_results['rho']
)

# Afficher les résultats
print(f"Prix Heston (PyTorch) pour K = {K_single}: {heston_price_single.item()}")
print(f"Prix Heston (NumPy) pour K = {K_single}: {old_heston_price_single}")