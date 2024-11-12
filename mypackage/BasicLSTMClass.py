from .PredictorClass import *
import torch
from sklearn.preprocessing import MinMaxScaler

class BasicLSTM(Predictor):
    """
    A simple LSTM-based predictor for stock prices. As long as the model takes as input array of prices and outputs a price this can be re-used
    """
    
    def __init__(self, test_data: MarketData, model_path: str):
        """
        Initializes the SimpleLSTM predictor.

        Args:
            test_data (MarketData): The test data used by the benchmark function.
            model_path (str): Path to the saved LSTM model.
        """
        super().__init__(test_data)  # Initialize the base class
        self.model = self.load_model(model_path)  # Load the trained model
    
    def load_model(self, model_path: str):
        """
        Load the trained LSTM model from the specified path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            LSTMModel: The loaded LSTM model.
        """
        model = torch.jit.load(model_path)  # Load the model
        model.eval()  # Set the model to evaluation mode
        return model

    def predict(self, sample: MarketData):

        prices_array = sample.get_sample_price(0)
        
        # Normalize the prices using the same scaler used during training
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices_array.reshape(-1, 1))  # Reshape for scaling, pytorch takes colomn vectors
        
        input = np.array([scaled_prices])
        input_tensor = torch.FloatTensor(input)  # Shape should be (1, seq_length, 1)
        
        # Make the prediction
        with torch.no_grad():  # No need to track gradients for inference, needed, don't remove.
            predicted = self.model(input_tensor)  # Get the output
        
        # Scale back to normal prices
        predicted_price = scaler.inverse_transform(predicted.numpy())
        predicted_price = predicted_price[0,0]  # Convert to a Python float

        # Here we only compute the increment, would need to find a way to include confidence. 
        # Percentage doesn't work generally since it depends on volatility and type of object (is 2% confident long or not?).
        increment = (predicted_price - prices_array[-1])

        if increment>=0:
            return 1
        else :
            return -1
        
