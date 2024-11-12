import numpy as np
from abc import ABC, abstractmethod
from .DataClass import MarketData

class Predictor(ABC):
    """
    Base class for stock market predictors. This class is meant to be used on a single "object"(ex : single stock) to give buy or sell signals.
    """

    def __init__(self, test_data: MarketData):
        """
            test_data (list or np.array): The test data used by the benchmark function must be handled by MarketData class.
        """
        self.test_data = test_data

    @abstractmethod
    def predict(self, sample: MarketData):
        """
        Predicts a score between -1 (confident short) and 1 (confident long) for the prices array of the given sample.
        Each subclass must implement this based on the input sample.
        Returns:
            float: A score between -1 (short) and 1 (long).
        """
        pass

    def backtest(self, backtest_horizons, data_subset: MarketData, buy_threshold=0.05, sell_threshold=-0.05):
        """
        Perform backtesting using the test data. Simulate predictions on past data and compare to actual results.
        Args:
            backtest_horizons (list or array): The specific time horizons to test the signal.
            data_subset (np.array): Data to use for backtesting. It has to be an array of arrays of prices (2d array)
            buy_threshold (float): The threshold above which the model signals a 'buy' (long position).
            sell_threshold (float): The threshold below which the model signals a 'sell' (short position).
        Returns:
            dict: Performance metrics for the backtesting (accuracy and returns per horizon).
        """

        results = {}
        for horizon in backtest_horizons:
            accuracy=0
            returns=0
            n=data_subset.sample_size()
            for i in range(n):
                n_accuracy, n_returns = self._backtest_horizon(data_subset.get_sample(i), horizon, buy_threshold, sell_threshold)
                accuracy += (1/n)*n_accuracy
                returns += (1/n)*n_returns
            results[horizon] = {"accuracy": accuracy, "returns": returns}

        return results

    def _backtest_horizon(self, sample: MarketData, horizon, buy_threshold, sell_threshold):
        """
        Backtest for a specific time horizon. It uses buy and sell tresholds so it doesn't take into account the confidence of the prediction
        Args:
            sample (np.array): must contain only one sample.(ex: only one prices array)
            horizon (int): The time horizon to test over.
            buy_threshold (float): The threshold for buy signals.
            sell_threshold (float): The threshold for sell signals.
        Returns:
            tuple: Accuracy and returns for the specific time horizon.
        """
        predictions = []
        actuals = []
        prices_array= sample.get_sample_price(0)
        n=prices_array.shape[0]
        for i in range(n - horizon):
            # Slice the test data up to the current point
            sample_slice = sample.data_slice(0,i+1)
            signal = self.predict(sample_slice)

            # Calculate future price change based on horizon
            future_price_change = (prices_array[i + horizon] - prices_array[i]) / prices_array[i]

            # Signal logic based on buy and sell thresholds
            if signal >= buy_threshold:
                prediction = 1  # Buy (Long)
            elif signal <= sell_threshold:
                prediction = 0  # Sell (Short)
            else:
                prediction = None  # Neutral, no trade made

            # Actual price movement (1 if price went up, 0 if it went down)
            actual = 1 if future_price_change > 0 else 0

            if prediction is not None:
                predictions.append(prediction)
                actuals.append(actual)

        # Calculate accuracy if there are any valid predictions
        if predictions:
            accuracy = np.mean([pred == act for pred, act in zip(predictions, actuals)])
        else:
            accuracy = None  # No predictions were made

        # Calculate returns based on predictions (long/short) if predictions exist
        if predictions:
            returns = sum(
                (1 if pred == 1 else -1) * ((prices_array[i + horizon] - prices_array[i]) / prices_array[i])
                for i, pred in enumerate(predictions)
            )
        else:
            returns = None  # No predictions, so no returns

        return accuracy, returns

    def benchmark(self):
        """
        Performs benchmarking on the model's performance. Each subclass can define its own benchmark strategy.
        By default, it uses backtesting on the test_data.
        Returns:
            dict: Performance results from backtesting or custom benchmark results.
        """
        return self.backtest(backtest_horizons=[1],data_subset=self.test_data)  # Default backtest with a list of horizons