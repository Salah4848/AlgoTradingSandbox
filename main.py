from mypackage import *
import yfinance as yf

def main():
    ticker = "AAPL"
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    prices = data["Close"].values
    #prices = [i for i in range(100)]

    market_data = MarketData(prices.T, "price_arrays")

    predictor = BasicMovingAverage(market_data)

    benchmark_results = predictor.benchmark(horizons=[1, 5, 10], plot_results=True)

    print("Benchmark Results:", benchmark_results)
if __name__=="__main__":
    main()
 