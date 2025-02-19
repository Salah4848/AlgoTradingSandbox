import pandas as pd
<<<<<<< HEAD


=======
##Function to get tickers
>>>>>>> 9eb2f471a7f3bc16450ad7d9490a8b4337bdb4f5
def get_top_tickers():
    # Fetch S&P 500 tickers
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)
    tickers = sp500_table[0]['Symbol'].tolist()
    return tickers
