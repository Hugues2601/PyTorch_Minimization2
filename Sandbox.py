from HestonModels.VanillaFormeAnalytique import *
from HestonModels.old_HestonModel import *
from config import CONFIG
import torch
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

# Charger le ticker du NASDAQ 100
ticker = yf.Ticker("^NDX")

# Afficher les dates d'expiration disponibles
expirations = ticker.options
print("Dates d'expiration disponibles:", expirations)

# Récupérer les options pour une date d'expiration spécifique
option_chain = ticker.option_chain(expirations[0])  # Première date d'expiration
calls = option_chain.calls  # Données des calls
puts = option_chain.puts    # Données des puts

# Afficher les premières lignes des DataFrames
print("Calls:")
print(calls.head())
print("\nPuts:")
print(puts.head())