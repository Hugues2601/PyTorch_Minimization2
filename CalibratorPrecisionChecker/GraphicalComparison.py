from config import CONFIG
from HestonModels.old_HestonModel import heston_price
from DataProcessing import data_processing

file, market_prices, T, IV, K = data_processing(CONFIG.df)
S0 = 603
r = 0.406
