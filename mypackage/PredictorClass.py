import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from .DataClass import MarketData

class Predictor(ABC):
    """
    Base class for stock market predictors. This class is meant to be used on a single "object" (e.g., single stock) to give buy or sell signals.
    """

    def __init__(self, test_data: MarketData):
        """
        Args:
            test_data (MarketData): The test data used by the benchmark function.
        """
        self.test_data = test_data

    @abstractmethod
    def predict(self, sample: MarketData) -> float:
        """
        Predicts a score between -1 (confident short) and 1 (confident long) for the prices array of the given sample.
        Each subclass must implement this based on the input sample.
        Returns:
            float: A score between -1 (short) and 1 (long).
        """
        pass

    def backtest(
        self,
        backtest_horizons: List[int],
        data_subset: MarketData,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
    ) -> Dict[int, Dict[str, float]]:
        """
        Perform backtesting using the test data. Simulate predictions on past data and compare to actual results.
        Args:
            backtest_horizons (List[int]): The specific time horizons to test the signal.
            data_subset (MarketData): Data to use for backtesting.
            buy_threshold (float): The threshold above which the model signals a 'buy' (long position).
            sell_threshold (float): The threshold below which the model signals a 'sell' (short position).
        Returns:
            Dict[int, Dict[str, float]]: Performance metrics for the backtesting (accuracy and returns per horizon).
        """
        # Input validation
        if not backtest_horizons:
            raise ValueError("backtest_horizons cannot be empty.")
        if buy_threshold <= sell_threshold:
            raise ValueError("buy_threshold must be greater than sell_threshold.")
        if data_subset.sample_size() == 0:
            raise ValueError("data_subset cannot be empty.")

        results = {}
        for horizon in backtest_horizons:
            if horizon <= 0:
                raise ValueError("Horizon must be a positive integer.")
            if horizon >= data_subset.get_sample_price(0).shape[0]:
                raise ValueError("Horizon exceeds the length of the price data.")

            accuracy = 0
            returns = 0
            n = data_subset.sample_size()
            for i in range(n):
                sample = data_subset.get_sample(i)
                n_accuracy, n_returns = self._backtest_horizon(sample, horizon, buy_threshold, sell_threshold)

                # Handle cases where no predictions are made
                if n_accuracy is not None:
                    accuracy += (1 / n) * n_accuracy
                if n_returns is not None:
                    returns += (1 / n) * n_returns

            results[horizon] = {"accuracy": accuracy, "returns": returns}

        return results

    def _backtest_horizon(
        self,
        sample: MarketData,
        horizon: int,
        buy_threshold: float,
        sell_threshold: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Backtest for a specific time horizon.
        Args:
            sample (MarketData): Must contain only one sample (e.g., only one prices array).
            horizon (int): The time horizon to test over.
            buy_threshold (float): The threshold for buy signals.
            sell_threshold (float): The threshold for sell signals.
        Returns:
            tuple: Accuracy and returns for the specific time horizon.
        """
        predictions = []
        actuals = []
        prices_array = sample.get_sample_price(0)
        n = prices_array.shape[0]
        for i in range(n - horizon):
            # Slice the test data up to the current point
            sample_slice = sample.data_slice(0, i + 1)
            signal = self.predict(sample_slice)

            # Calculate future price change based on horizon
            future_price_change = prices_array[i + horizon] - prices_array[i]
            # Signal logic based on buy and sell thresholds
            if signal >= buy_threshold:
                prediction = 1  # Buy (Long)
            elif signal <= sell_threshold:
                prediction = -1  # Sell (Short)
            else:
                prediction = 0  # Neutral, no trade made

            # Actual price movement (1 if price went up, 0 if it went down)
            actual = 1 if future_price_change > 1 else -1 # MIGHT DO SOMETHING ELSE HERE

            if prediction is not None:
                predictions.append(prediction)
                actuals.append(actual)

        # Calculate accuracy if there are any valid predictions
        if predictions:
            accuracy = np.mean([pred == act for pred, act in zip(predictions, actuals) if pred!=0])
        else:
            accuracy = None  # No predictions were made

        # Calculate returns based on predictions (long/short) if predictions exist
        if predictions:
            returns = sum(
                pred * (prices_array[i + horizon] - prices_array[i])
                for i, pred in enumerate(predictions)
            )
        else:
            returns = None  # No predictions, so no returns

        return accuracy, returns

    def benchmark(
        self,
        horizons: List[int] = [1, 5, 10],
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
        plot_results: bool = True,
    ) -> Dict[str, Dict]:
        """
        Performs benchmarking on the model's performance. Includes aggregate metrics and comparisons to a buy-and-hold strategy.
        Args:
            horizons (List[int]): Time horizons to test the strategy.
            buy_threshold (float): Threshold for buy signals.
            sell_threshold (float): Threshold for sell signals.
            plot_results (bool): Whether to plot performance visualizations.
        Returns:
            Dict[str, Dict]: A dictionary containing performance metrics.
        """
        # Perform backtesting
        backtest_results = self.backtest(
            backtest_horizons=horizons,
            data_subset=self.test_data,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        # Calculate buy-and-hold performance
        buy_and_hold_returns = self._calculate_buy_and_hold_returns()
        buy_and_hold_avg_return = np.mean(buy_and_hold_returns)
        buy_and_hold_sharpe = buy_and_hold_avg_return / (np.std(buy_and_hold_returns) + 1e-10)
        buy_and_hold_max_drawdown = self._calculate_max_drawdown(buy_and_hold_returns)
        buy_and_hold_win_rate = np.mean(np.array(buy_and_hold_returns) > 0)

        # Calculate performance metrics for each horizon
        performance_metrics = {}
        for horizon, results in backtest_results.items():
            avg_return = results["returns"]
            accuracy = results["accuracy"]

            # Calculate Sharpe Ratio (assuming risk-free rate = 0)
            sharpe_ratio = avg_return / (np.std(buy_and_hold_returns) + 1e-10)  # Use buy-and-hold std for consistency

            # Calculate Win Rate (approximate, since we don't have individual trade results)
            win_rate = accuracy  # Use accuracy as a proxy for win rate

            # Store metrics
            performance_metrics[f"Horizon {horizon}"] = {
                "Accuracy": round(accuracy, 2),
                "Avg Return": round(avg_return, 2),
                "Sharpe Ratio": round(sharpe_ratio, 2),
                "Win Rate": round(win_rate, 2),
            }

        # Add buy-and-hold metrics
        performance_metrics["Buy-and-Hold"] = {
            "Avg Return": round(buy_and_hold_avg_return, 2),
            "Sharpe Ratio": round(buy_and_hold_sharpe, 2),
            "Max Drawdown": round(buy_and_hold_max_drawdown, 2),
            "Win Rate": round(buy_and_hold_win_rate, 2),
        }

        # Print results in a clean format
        print("Benchmark Results:")
        for key, metrics in performance_metrics.items():
            print(f"\n{key}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

        # Plot results if requested
        if plot_results:
            self._plot_performance(backtest_results, buy_and_hold_returns)

        return performance_metrics

    def _plot_performance(self, backtest_results: Dict[int, Dict], buy_and_hold_returns: List[float]):
        """
        Plots the performance of the strategy versus the buy-and-hold baseline.
        Args:
            backtest_results (Dict[int, Dict]): Backtest results for the strategy.
            buy_and_hold_returns (List[float]): Returns for the buy-and-hold strategy.
        """
        horizons = list(backtest_results.keys())
        strategy_returns = [backtest_results[horizon]["returns"] for horizon in horizons]
        buy_and_hold_avg_return = np.mean(buy_and_hold_returns)

        plt.figure(figsize=(10, 6))
        plt.bar(horizons, strategy_returns, label="Strategy")
        plt.axhline(y=buy_and_hold_avg_return, color="red", linestyle="--", label="Buy-and-Hold")
        plt.title("Average Returns: Strategy vs Buy-and-Hold")
        plt.xlabel("Horizon")
        plt.ylabel("Average Returns")
        plt.legend()
        plt.grid()
        plt.show()

    def _calculate_buy_and_hold_returns(self) -> List[float]:
        """
        Calculates the returns for a buy-and-hold strategy.
        Returns:
            List[float]: Returns for the buy-and-hold strategy.
        """
        prices = self.test_data.get_sample_price(0)
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        return returns

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """
        Calculates the maximum drawdown for a given series of returns.
        Args:
            returns (List[float]): Series of returns.
        Returns:
            float: Maximum drawdown.
        """
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-10)
        return np.max(drawdown)