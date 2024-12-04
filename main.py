from DataProcessing import data_processing, api_retriever, data_to_csv, get_treasury_yield
from HestonModels.VanillaFormeAnalytique import heston_price
from Calibrator import calibrate
from config import CONFIG
import torch
import numpy as np
from results import write_results_to_file

def run():

    file, market_prices, T, IV, K = data_processing(CONFIG.df)
    initial_guess = CONFIG.initial_guess
    S0 = 603
    r = 0.0406
    calibrated_params = calibrate(S0, market_prices, K, T, r, initial_guess)
    print("Calibrated Parameters:")
    print(calibrated_params)
    write_results_to_file(calibrated_params)


if __name__ == "__main__":
    run()

