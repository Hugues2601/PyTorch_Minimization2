from config import CONFIG
from HestonModels.old_HestonModel import heston_price
from DataProcessing import data_processing
import matplotlib.pyplot as plt

file, market_prices, T, IV, K = data_processing(CONFIG.df)
S0 = 603
r = 0.406
kappa, v0, theta, sigma, rho = 1.911, 0.0095, 0.0099, 0.2893, -0.5889

heston_prices = []
for i in range(len(K)):
    price = heston_price(S0, K[i], T[i], r, kappa, v0, theta, sigma, rho)
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