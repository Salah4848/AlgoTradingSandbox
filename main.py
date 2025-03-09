from mypackage import *
import yfinance as yf
def main():
    ticker = yf.Ticker("AAPL")
    data =ticker.history(start='2020-01-01', end=None) #'2020-01-01'
    #prices = [i for i in range(100)]
    market_data = MarketData([data], "yfinance_Ticker_method_dfs")

    predictor = MACDPredictor(market_data)
    benchmark_results = predictor.benchmark(horizons=[1,7,15], plot_results=True)

    print("Benchmark Results:", benchmark_results)
if __name__=="__main__":
    main()
 

 