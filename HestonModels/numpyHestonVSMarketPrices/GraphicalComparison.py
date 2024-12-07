from results import Results
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from HestonModels.old_HestonModel import old_heston_price
from DataProcessing import data_processing
from config import CONFIG

def numpy_heston_v_market_prices():
    file, market_prices, T, IV, K = data_processing(CONFIG.symbol)
    calibrated_params = Results.calibration_results
    S0 = CONFIG.S0
    r = CONFIG.r
    kappa = calibrated_params['kappa']
    v0 = calibrated_params['v0']
    theta = calibrated_params['theta']
    sigma = calibrated_params['sigma']
    rho = calibrated_params['rho']

    # Group data by unique maturities (T)
    unique_maturities = sorted(set(T))
    grouped_data = defaultdict(list)

    for j in range(len(T)):
        grouped_data[T[j]].append((K[j], market_prices[j]))

    # Create one plot per T
    for t in unique_maturities:
        strikes = [k for k, _ in grouped_data[t]]
        market = [p for _, p in grouped_data[t]]
        heston = [
            old_heston_price(S0, k, t, r, kappa, v0, theta, sigma, rho)
            for k in strikes
        ]

        # Plot for this maturity
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, market, label='Market Prices', marker='o', linestyle='-', color='blue')
        plt.plot(strikes, heston, label='Heston Model Prices', marker='x', linestyle='--', color='orange')
        plt.xlabel('Strike Price (K)')
        plt.ylabel('Option Price')
        plt.title(f'Comparison of Heston Model Prices and Market Prices for T = {t:.2f}')
        plt.legend()
        plt.grid(True)
        plt.show()
