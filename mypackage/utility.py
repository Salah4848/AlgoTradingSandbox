import pandas as pd
##Function to get tickers
def get_top_tickers():
    # Fetch S&P 500 tickers
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)
    tickers = sp500_table[0]['Symbol'].tolist()
    return tickers
