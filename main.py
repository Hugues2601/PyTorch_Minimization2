from DataProcessing import data_processing, api_retriever, data_to_csv, get_treasury_yield
from HestonModels.VanillaFormeAnalytique import heston_price
from HestonModels.old_HestonModel import old_heston_price
from Calibrator import calibrate
from config import CONFIG
from results import Results
import torch
import numpy as np
from results import write_results_to_file
from collections import defaultdict

def run():

    file, market_prices, T, IV, K = data_processing(CONFIG.df)
    initial_guess = CONFIG.initial_guess
    S0 = 603
    r = 0.0406
    # calibrated_params = calibrate(S0, market_prices, K, T, r, initial_guess, max_epochs=1000)
    # print("Calibrated Parameters:")
    # print(calibrated_params)
    # write_results_to_file(calibrated_params)
    calibrated_params = {'kappa': 1.8760051842660947, 'v0': 0.015424479825010152, 'theta': 0.02359564977565198, 'sigma': 0.18289090205926006, 'rho': -0.9298662939805773}

    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming calibrated_params is a dictionary with the required parameters
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



if __name__ == "__main__":
    run()

