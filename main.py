from DataProcessing import data_processing, api_retriever, data_to_csv, get_treasury_yield
from HestonModels.VanillaFormeAnalytique import heston_price
from HestonModels.numpyHestonVSMarketPrices.GraphicalComparison import numpy_heston_v_market_prices
from Calibrator import calibrate
from config import CONFIG
from results import Results
import torch
import numpy as np
from results import write_results_to_file
from collections import defaultdict
import matplotlib.pyplot as plt

def run():
    file, market_prices, T, IV, K = data_processing(CONFIG.df)
    print(f"Nb of options :{len(market_prices)} | mean price: {np.mean(market_prices)}")
    initial_guess = CONFIG.initial_guess
    calibrated_params = calibrate(CONFIG.S0, market_prices, K, T, CONFIG.r, initial_guess, max_epochs=1000, loss_threshold=2)
    print("Calibrated Parameters:")
    print(calibrated_params)
    write_results_to_file(calibrated_params)
    numpy_heston_v_market_prices()


if __name__ == "__main__":
    plt.close('all')
    run()

