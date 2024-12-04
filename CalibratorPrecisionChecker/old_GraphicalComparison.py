from config import CONFIG
from HestonModels.old_HestonModel import old_heston_price, old_heston_price
from DataProcessing import data_processing
import matplotlib.pyplot as plt
import torch
from HestonModels.VanillaFormeAnalytique import heston_price

file, market_prices, T, IV, K = data_processing(CONFIG.df)
S0 = 603
r = 0.406
kappa, v0, theta, sigma, rho = 1.9204, 0.01928, 0.193820, 0.280299, -0.580135

heston_prices = []
for i in range(len(K)):
    price = old_heston_price(S0, K[i], T[i], r, kappa, v0, theta, sigma, rho)
    heston_prices.append(price.item())  # Convertir en float si nécessaire

# Organiser les données par maturité
data_by_maturity = {}
for i in range(len(T)):
    maturity = T[i]
    if maturity not in data_by_maturity:
        data_by_maturity[maturity] = {"market": [], "heston": [], "strikes": []}
    data_by_maturity[maturity]["market"].append(market_prices[i])
    data_by_maturity[maturity]["heston"].append(heston_prices[i])
    data_by_maturity[maturity]["strikes"].append(K[i])

# Tracer les graphiques
for maturity, data in data_by_maturity.items():
    plt.figure()
    plt.scatter(data["market"], data["heston"], label="Heston vs Market")
    plt.plot([min(data["market"]), max(data["market"])],
             [min(data["market"]), max(data["market"])],
             color="red", linestyle="--", label="Perfect Fit")
    plt.title(f"Heston Prices vs Market Prices (Maturity = {maturity})")
    plt.xlabel("Market Prices")
    plt.ylabel("Heston Prices")
    plt.legend()
    plt.grid(True)
    plt.show()



# ############### Price old heston vs new heston
# S0 = torch.tensor(207, device=CONFIG.device)  # Prix spot
# r = torch.tensor(0.0406, device=CONFIG.device)   # Taux sans risque
# kappa, v0, theta, sigma, rho = (
#     torch.tensor(1.965, device=CONFIG.device, requires_grad=True),
#     torch.tensor(0.0779, device=CONFIG.device, requires_grad=True),
#     torch.tensor(0.104285, device=CONFIG.device, requires_grad=True),
#     torch.tensor(0.229545, device=CONFIG.device, requires_grad=True),
#     torch.tensor(-0.53984, device=CONFIG.device, requires_grad=True)
# )  # Paramètres calibrés
#
# market_prices = torch.tensor(market_prices, device=CONFIG.device)
# K = torch.tensor(K, device=CONFIG.device)
# T = torch.tensor(T, device=CONFIG.device)
#
#
# heston_prices = heston_price(S0, K, T, r, kappa, v0, theta, sigma, rho)
#
# # Organiser les données par maturité
# data_by_maturity = {}
# for i in range(len(T)):
#     maturity = T[i].item()  # Convertir en float pour les clés
#     if maturity not in data_by_maturity:
#         data_by_maturity[maturity] = {"market": [], "heston": []}
#     data_by_maturity[maturity]["market"].append(market_prices[i].item())
#     data_by_maturity[maturity]["heston"].append(heston_prices[i].item())
#
# # Tracer un graphique unique
# plt.figure(figsize=(10, 6))
# colors = ["blue", "green", "orange", "purple", "brown"]  # Palette pour les maturités
#
# for i, (maturity, data) in enumerate(data_by_maturity.items()):
#     plt.scatter(data["market"], data["heston"], label=f"Maturity = {maturity}", color=colors[i % len(colors)])
#
# # Ajouter une ligne de référence (Perfect Fit)
# all_prices = list(market_prices.cpu().numpy()) + list(heston_prices.cpu().detach().numpy())
# plt.plot([min(all_prices), max(all_prices)], [min(all_prices), max(all_prices)],
#          color="red", linestyle="--", label="Perfect Fit")
#
# # Personnalisation
# plt.title("Heston Prices vs Market Prices")
# plt.xlabel("Market Prices")
# plt.ylabel("Heston Prices")
# plt.legend()
# plt.grid(True)
# plt.show()