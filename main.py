from DataProcessing import data_processing, api_retriever, data_to_csv, get_treasury_yield
from HestonModels.VanillaFormeAnalytique import heston_price
from Calibrator import calibrate
from config import CONFIG
import torch
import numpy as np

def run():
    pass


if __name__ == "__main__":
    # # yield_10y = get_treasury_yield()
    #
    # Example data
    file, market_prices, T, IV, K = data_processing(CONFIG.df)
    print(f"nb d'options : {len(market_prices)}")
    S0 = 603
    r = 0.0406

    # Initial guesses for parameters
    initial_guess = {
        'kappa': 2.0,
        'v0': 0.1,
        'theta': 0.1,
        'sigma': 0.2,
        'rho': -0.5
    }

    # Calibrate the model
    calibrated_params = calibrate(S0, market_prices, K, T, r, initial_guess)

    print("Calibrated Parameters:")
    print(calibrated_params)

