import pandas as pd
from datetime import datetime

class CONFIG:
    device = "cuda"

    df = pd.read_csv("C:\\Users\\hugue\\Desktop\\Master Thesis\\Data\\20241202_SPY_calls.csv")


    # Alpha vantage options retrieving
    apikey = "UC2NJBUMF1FT83J1"
    symbol = "SPY"
    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={apikey}"

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

