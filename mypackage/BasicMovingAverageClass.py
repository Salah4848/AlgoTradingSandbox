from .PredictorClass import *




class BasicMovingAverage(Predictor):
    """
    A predictor based on moving average crossover.
    """
    def __init__(self, test_data, short_window=5, long_window=15):
        """
            test_data (list or np.array): The test data used by the benchmark function. It must be array of array of prices.
            Compares the long moving average mean given by long_window and short moving average mean given by short_window to make a trade.
        """
        super().__init__(test_data)
        self.short_window = short_window
        self.long_window = long_window

    def predict(self, sample: MarketData):

        if sample.sample_size()>1:
            print("Warning: imput data must have sample size 1")
        prices = sample.get_sample_price(0)

        # Check if there's enough data for the moving averages
        if len(prices) < self.long_window:
            return 0  # Neutral if insufficient data

        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])

        # Signal generation based on short vs long MA crossover
        if short_ma > long_ma:
            return 1.0  # Confident long
        elif short_ma < long_ma:
            return -1.0  # Confident short
        else:
            return 0.0  # Neutral