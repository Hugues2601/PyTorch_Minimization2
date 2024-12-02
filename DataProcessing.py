from datetime import datetime
import pandas as pd
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