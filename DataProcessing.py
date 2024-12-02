from datetime import datetime
import pandas as pd
import requests
import json
from config import CONFIG
import pandas as pd
import os

def api_retriever():
    response = requests.get(CONFIG.url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        flattened_df = pd.DataFrame([row['data'] for _, row in df.iterrows()])
        return flattened_df

    else:
        print(f"Erreur lors de la requête : HTTP {response.status_code}")



def data_to_csv(df):
    # Garder seulement les colonnes nécessaires dans le premier DataFrame
    options_df = df[['last', 'expiration', 'implied_volatility', 'strike', 'open_interest', 'type']]

    # Créer un second DataFrame avec les colonnes delta, gamma, theta, vega et rho
    greeks_df = df[['delta', 'gamma', 'theta', 'vega', 'rho']]

    options_df_calls = options_df[options_df['type']=="call"]
    options_df_puts = options_df[options_df['type']=="put"]

    os.makedirs(CONFIG.save_path, exist_ok=True)

    greeks_df.to_csv(os.path.join(CONFIG.save_path, f"{CONFIG.todaydate}_{CONFIG.apikey}_greeks_df.csv"), index=False)
    options_df_calls.to_csv(os.path.join(CONFIG.save_path, f"{CONFIG.todaydate}_{CONFIG.apikey}_calls.csv"), index=False)
    options_df_calls.to_csv(os.path.join(CONFIG.save_path, f"{CONFIG.todaydate}_{CONFIG.apikey}_put.csv"), index=False)


def data_processing(df):

    df['expiration'] = pd.to_datetime(df['expiration'])


    today = datetime.today()
    df['TimeToMaturity'] = df['expiration'].apply(lambda x: (x - today).days / 365)
    df = df[(df["implied_volatility"] < 0.8) & (df['last'] > 2) & (df["open_interest"] > 10) & (df["TimeToMaturity"]>0.2)]
    df = df.reset_index(drop=True)
    mp = df.iloc[:, 0].tolist()
    T = df.iloc[:, -1].tolist()
    IV = df.iloc[:, 2].tolist()
    K = df.iloc[:, 3].tolist()

    return df, mp, T, IV, K