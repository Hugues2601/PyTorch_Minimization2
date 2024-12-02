import pandas as pd
from datetime import datetime

class CONFIG:
    device = "cuda"

    df = pd.read_csv("C:\\Users\\hugue\\Desktop\\Master Thesis\\Data\\filtered_options_20252026.csv")

    apikey = "UC2NJBUMF1FT83J1"
    symbol = "SPY"
    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={apikey}"


    save_path = r"C:\Users\hugue\Desktop\Master Thesis\Data"
    todaydate = (datetime.now()).strftime("%Y%m%d")

