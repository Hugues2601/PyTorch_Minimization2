import pandas as pd

class CONFIG:
    device = "cuda"

    df = pd.read_csv("C:\\Users\\hugue\\Desktop\\Master Thesis\\Data\\filtered_options_20252026.csv")

    apikey = "UC2NJBUMF1FT83J1"
    symbol = "AMZN"
    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={apikey}"

