from DataProcessing import data_processing, api_retriever, data_to_csv
from HestonModels.VanillaFormeAnalytique import heston_price
from Calibrator import calibrate
from config import CONFIG
import torch

def run():
    file = data_processing(CONFIG.df)
    print(file)

if __name__ == "__main__":


    # Example data
    file, market_prices, T, IV, K = data_processing(CONFIG.df)
    S0 = 603
    # market_prices = [10.0, 12.0, 8.0]  # Example market prices
    # K = [90.0, 100.0, 110.0]  # Corresponding strikes
    # T = [1.0, 1.0, 1.0]  # Corresponding maturities
    r = 0.05  # Risk-free rate

    # Initial guesses for parameters
    initial_guess = {
        'kappa': 2.0,
        'v0': 0.1,
        'theta': 0.1,
        'sigma': 0.2,
        'rho': -0.5
    }

    # Calibrate the model
    calibrated_params = calibrate(S0, market_prices, K, T, r, initial_guess, epochs=30, lr=0.01)

    print("Calibrated Parameters:")
    print(calibrated_params)

