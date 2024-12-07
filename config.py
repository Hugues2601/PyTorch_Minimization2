import pandas as pd
from datetime import datetime

class CONFIG:
    device = "cuda"

    df = pd.read_csv("C:\\Users\\hugue\\Desktop\\Master Thesis\\Data\\20241202_SPY_calls.csv")


    # Alpha vantage options retrieving
    apikey = "UC2NJBUMF1FT83J1"
    apikey_polygon = "yQyMsqx8zbeR_cAgp_QNx92EGsv6Atcl"
    symbol = "SPY"
    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={apikey}"
    url_polygon = "https://api.polygon.io/v3/trades/options"

    # treasury yield retrieving parameters
    url_t_yield = f"https://www.alphavantage.co/query"
    params_t_yield = {
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": "10Y",
        "apikey": apikey
    }

    save_path = r"C:\Users\hugue\Desktop\Master Thesis\Data"
    todaydate = (datetime.now()).strftime("%Y%m%d")

    # Heston Vanilla model initial guess :
    initial_guess = {
        'kappa': 2.0,
        'v0': 0.1,
        'theta': 0.1,
        'sigma': 0.2,
        'rho': -0.5
    }

