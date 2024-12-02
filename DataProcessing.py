from datetime import datetime
import pandas as pd
import requests
import json
from config import CONFIG

def api_retriever():
    response = requests.get(CONFIG.url)
    # Vérifier le statut de la requête
    if response.status_code == 200:
        # Analyser les données JSON
        data = response.json()
        print(data)
    else:
        print(f"Erreur lors de la requête : HTTP {response.status_code}")

def data_processing(df):

    df['expiration'] = pd.to_datetime(df['expiration'])

    # Calculer le temps restant en années
    today = datetime.today()
    df['TimeToMaturity'] = df['expiration'].apply(lambda x: (x - today).days / 365)
    mp = df.iloc[:, 0].tolist()
    T = df.iloc[:, -1].tolist()
    IV = df.iloc[:, 2].tolist()
    K = df.iloc[:, 3].tolist()

    return df, mp, T, IV, K