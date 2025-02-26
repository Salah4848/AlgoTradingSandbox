from .PredictorClass import *
from .indicators import *
import joblib

class BasicTreeIndicators(Predictor):
    """
    A simple LSTM-based predictor for stock prices. As long as the model takes as input array of prices and outputs a price this can be re-used
    """
    
    def __init__(self, test_data: MarketData, model_path: str, numdays: int =10):
        """
        Args:
            test_data (MarketData): The test data used by the benchmark function.
            model_path (str): Path to the saved LSTM model.
        """
        super().__init__(test_data)  # Initialize the base class
        self.model = self.load_model(model_path)  # Load the trained model
        self.numdays = numdays
    
    def load_model(self, model_path: str):
        model = joblib.load(model_path)  # Load the model
        return model

    def predict(self, sample: MarketData):
        if sample.sample_length() < 70: # This is needed to compute the indicators
            return 0

        opens = sample.get_sample_open(0)
        highs = sample.get_sample_high(0)
        lows = sample.get_sample_low(0)
        closes = sample.get_sample_close(0)
        volumes = sample.get_sample_volume(0)
        dividends = sample.get_sample_dividends(0)
        stockplits = sample.get_sample_stocksplits(0)
        
        indicatorsdf = calculate_technical_indicators(highs,lows,closes,volumes)
        indicators = indicatorsdf.to_numpy()
        indicators = indicators.T

        datatot = np.concatenate(([opens,highs,lows,closes,volumes,dividends,stockplits],indicators))
        dataslice = datatot[:,-self.numdays:]
        input = np.array([dataslice.flatten('F')])

        out = self.model.predict(input)

        signal = 2*out[0]-1

        return signal