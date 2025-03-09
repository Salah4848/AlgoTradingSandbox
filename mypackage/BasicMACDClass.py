from .PredictorClass import *

class MACDPredictor(Predictor):
    """
    Simple MACD indicator to generate singals.
    """

    def __init__(self, test_data: MarketData, short_window=12, long_window=26, signal_window=9):
        """
        We use the recommended 26/11/9 configuration as default (source: Investopedia). Prediction horizon is a 1 day in general
        Args:
            test_data (MarketData): Data used for testing/backtesting.
            short_window (int): Window length for the short EMA (default 12).
            long_window (int): Window length for the long EMA (default 26).
            signal_window (int): Window length for the signal line EMA (default 9).
        """
        super().__init__(test_data)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def calculate_ema(self, data, window):
        """
        Calculates the Exponential Moving Average (EMA) of a 1D array in the specified window.

        """
        ema = np.zeros(len(data))
        smoothing = 2 #Recommended on Investopedia
        alpha = smoothing / (window + 1)
        ema[0] = data[0]  # Start EMA from first data point
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def predict(self, sample: MarketData):
        """
        Uses MACD to generate a prediction score for a given sample.

        """
        if sample.sample_size()>1:
            print("Warning: imput data must have sample size 1")
        prices = sample.get_sample_price(0)

        if len(prices) < 3*self.long_window*self.signal_window: # Neutral if insufficient data, this should guarentee we have enough data.
            return 0

        # Calculate MACD and Signal Line
        short_ema = self.calculate_ema(prices, self.short_window)
        long_ema = self.calculate_ema(prices, self.long_window)
        macd_line = short_ema - long_ema
        signal_line = self.calculate_ema(macd_line, self.signal_window)

        # Use the most recent MACD and signal line values for prediction
        macd_diff = macd_line[-1] - signal_line[-1]
        macd_diffold = macd_line[-2] - signal_line[-2]

        # Map macd_diff to a range of -1 to 1 for prediction score
        if macd_diff>0 and macd_diffold<0:
            return 1
        if macd_diff<0 and macd_diffold>0:
            return -1
        else:
            return 0
