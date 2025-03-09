import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
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
        position_size: float = 1.0,
    ) -> Dict[int, Dict[str, float]]:
        """
        Perform backtesting using the test data. Simulate predictions on past data and compare to actual results.
        EACH SAMPLE MUST HAVE THE SAME LENGTH.
        
        Args:
            backtest_horizons (List[int]): The specific time horizons to test the signal.
            data_subset (MarketData): Data to use for backtesting.
            buy_threshold (float): The threshold above which the model signals a 'buy' (long position).
            sell_threshold (float): The threshold below which the model signals a 'sell' (short position).
            position_size (float): Size of each position as a fraction of capital (default: 1.0).
            
        Returns:
            Dict[int, Dict[str, float]]: Performance metrics for the backtesting.
        """
        # Input validation
        if not backtest_horizons:
            raise ValueError("backtest_horizons cannot be empty.")
        if buy_threshold <= sell_threshold:
            raise ValueError("buy_threshold must be greater than sell_threshold.")
        if data_subset.sample_size() == 0:
            raise ValueError("data_subset cannot be empty.")
        if not (0 < position_size <= 1.0):
            raise ValueError("position_size must be between 0 and 1.")

        results = {}
        for horizon in backtest_horizons:
            if horizon <= 0:
                raise ValueError("Horizon must be a positive integer.")
            
            # Collect all backtest results for this horizon
            all_trades = []
            all_returns = []
            equity = 1.0     # Start with $1 of capital
            avg_equity_curve = None
            
            n = data_subset.sample_size()
            for i in range(n):
    
                sample = data_subset.get_sample(i)
                trades, returns, equity_curve = self._backtest_horizon(
                    sample, horizon, buy_threshold, sell_threshold, position_size,equity
                )
                if i==0:
                    avg_equity_curve = (1/n)*np.array(equity_curve)
                else:
                    avg_equity_curve += (1/n)*np.array(equity_curve)

                if trades:
                    all_trades.extend(trades)
                    all_returns.extend(returns)
                    
                    # Adjust the P&L curve by adding the new returns
                    # (scaled to start from the last point)
            
            # Calculate performance metrics
            if all_trades:
                win_rate = sum(ret > 0 for ret in all_returns) / len(all_returns)
                avg_return = np.mean(all_returns)
                total_return = np.prod(1 + np.array(all_returns)) - 1
                
                # Risk metrics
                std_returns = np.std(all_returns)
                sharpe = avg_return / (std_returns + 1e-10)
                
                # Drawdown analysis
                max_drawdown = self._calculate_max_drawdown(avg_equity_curve)
                
                # For accuracy, we only consider directional trades (long or short)
                directional_trades = [(pred, ret) for pred, ret in all_trades if pred != 0]
                if directional_trades:
                    accuracy = sum(np.sign(pred) == np.sign(ret) for pred, ret in directional_trades) / len(directional_trades)
                else:
                    accuracy = None
            else:
                win_rate = None
                avg_return = None
                total_return = None
                sharpe = None
                max_drawdown = None
                accuracy = None
            
            results[horizon] = {
                "win_rate": win_rate,
                "accuracy": accuracy,
                "avg_return": avg_return,
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "num_trades": len(all_trades),
                "equity_curve": avg_equity_curve,
            }

        return results

    def _backtest_horizon(
        self,
        sample: MarketData,
        horizon: int,
        buy_threshold: float,
        sell_threshold: float,
        position_size: float = 1.0,
        equity: float = 1.0
    ) -> Tuple[List[Tuple[int, float]], List[float], List[float]]:
        """
        Backtest for a specific time horizon.
        
        Args:
            sample (MarketData): Must contain only one sample (e.g., only one prices array).
            horizon (int): The time horizon to test over.
            buy_threshold (float): The threshold for buy signals.
            sell_threshold (float): The threshold for sell signals.
            position_size (float): Size of each position as a fraction of capital.
            
        Returns:
            tuple: 
                - List of (prediction, actual_return) pairs
                - List of trade returns
                - Equity curve
        """
        trades = []
        returns = []
        equity_curve = [equity]  # Start with specified equity
        
        prices_array = sample.get_sample_price(0)
        if horizon >= prices_array.shape[0]:
            return [], [], [1.0]  # Return empty lists if horizon exceeds data length
            
        n = prices_array.shape[0]
        
        # Track current position (1 for long, -1 for short, 0 for flat)
        current_position = 0
        entry_price = 0
        
        for i in range(n - horizon):
            equity_update = equity_curve[-1]

            # Slice the test data up to the current point
            sample_slice = sample.data_slice(0, i + 1)
            signal = self.predict(sample_slice)
            
            # Current price and future price
            current_price = prices_array[i]
            future_price = prices_array[i + horizon]
            
            # Determine trading action based on signal and current position
            if current_position == 0:  # Currently flat
                if signal >= buy_threshold:
                    # Enter long position
                    current_position = 1
                    entry_price = current_price
                elif signal <= sell_threshold:
                    # Enter short position
                    current_position = -1
                    entry_price = current_price
            elif current_position == 1:  # Currently long
                if signal <= sell_threshold:
                    # Close long and possibly go short
                    # Calculate return from long position
                    position_return = (current_price - entry_price) / entry_price * position_size
                    returns.append(position_return)
                    trades.append((1, position_return))
                    
                    # Update equity
                    equity_update=equity_curve[-1] * (1 + position_return)
                    
                    # Enter short position
                    current_position = -1
                    entry_price = current_price
                # Else maintain long position
            elif current_position == -1:  # Currently short
                if signal >= buy_threshold:
                    # Close short and possibly go long
                    # Calculate return from short position
                    position_return = (entry_price - current_price) / entry_price * position_size
                    returns.append(position_return)
                    trades.append((-1, position_return))
                    
                    # Update equity
                    equity_update=equity_curve[-1] * (1 + position_return)
                    
                    # Enter long position
                    current_position = 1
                    entry_price = current_price
                # Else maintain short position
            
            # Close any remaining position at the end of the backtest
            if i == n - horizon - 1 and current_position != 0:
                if current_position == 1:
                    position_return = (future_price - entry_price) / entry_price * position_size
                    returns.append(position_return)
                    trades.append((1, position_return))
                else:  # current_position == -1
                    position_return = (entry_price - future_price) / entry_price * position_size
                    returns.append(position_return)
                    trades.append((-1, position_return))
                
                # Update equity
                equity_update=equity_curve[-1] * (1 + position_return)
            equity_curve.append(equity_update)
            

        
        return trades, returns, equity_curve

    def benchmark(
        self,
        horizons: List[int] = [1, 5, 10],
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
        position_size: float = 1.0,
        plot_results: bool = True,
    ) -> Dict[str, Dict]:
        """
        Performs benchmarking on the model's performance. Includes aggregate metrics and comparisons to a buy-and-hold strategy.
        
        Args:
            horizons (List[int]): Time horizons to test the strategy.
            buy_threshold (float): Threshold for buy signals.
            sell_threshold (float): Threshold for sell signals.
            position_size (float): Size of each position as a fraction of capital.
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
            position_size=position_size,
        )

        # Calculate buy-and-hold performance
        prices = self.test_data.get_sample_price(0)
        buy_and_hold_total_return = (prices[-1] - prices[0]) / prices[0]
        buy_and_hold_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        buy_and_hold_avg_return = np.mean(buy_and_hold_returns)
        buy_and_hold_std = np.std(buy_and_hold_returns)
        buy_and_hold_sharpe = buy_and_hold_avg_return / (buy_and_hold_std + 1e-10)
        buy_and_hold_max_drawdown = self._calculate_max_drawdown(np.cumprod(1 + np.array(buy_and_hold_returns)))
        buy_and_hold_win_rate = np.mean(np.array(buy_and_hold_returns) > 0)

        # Prepare performance metrics
        performance_metrics = {}
        
        # Add buy-and-hold metrics
        performance_metrics["Buy-and-Hold"] = {
            "Total Return": round(buy_and_hold_total_return * 100, 2),
            "Avg Return (%)": round(buy_and_hold_avg_return * 100, 2),
            "Std Dev (%)": round(buy_and_hold_std * 100, 2),
            "Sharpe Ratio": round(buy_and_hold_sharpe, 2),
            "Max Drawdown (%)": round(buy_and_hold_max_drawdown * 100, 2),
            "Win Rate (%)": round(buy_and_hold_win_rate * 100, 2),
        }

        # Add strategy metrics for each horizon
        for horizon, results in backtest_results.items():
            if results["num_trades"] > 0:
                metrics = {
                    "Total Return (%)": round(results["total_return"] * 100, 2) if results["total_return"] is not None else "N/A",
                    "Avg Return (%)": round(results["avg_return"] * 100, 2) if results["avg_return"] is not None else "N/A",
                    "Sharpe Ratio": round(results["sharpe_ratio"], 2) if results["sharpe_ratio"] is not None else "N/A",
                    "Max Drawdown (%)": round(results["max_drawdown"] * 100, 2) if results["max_drawdown"] is not None else "N/A",
                    "Win Rate (%)": round(results["win_rate"] * 100, 2) if results["win_rate"] is not None else "N/A",
                    "Accuracy (%)": round(results["accuracy"] * 100, 2) if results["accuracy"] is not None else "N/A",
                    "Number of Trades": results["num_trades"],
                }
            else:
                metrics = {"Number of Trades": 0, "Note": "No trades generated for this horizon"}
                
            performance_metrics[f"Strategy (Horizon {horizon})"] = metrics

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
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Performance Analysis', fontsize=16)
        
        # Plot 1: Returns by horizon
        horizons = list(backtest_results.keys())
        total_returns = [backtest_results[h]["total_return"] * 100 if backtest_results[h]["total_return"] is not None else 0 
                         for h in horizons]
        buy_and_hold_total = (np.prod(1 + np.array(buy_and_hold_returns)) - 1) * 100
        
        axs[0, 0].bar(horizons, total_returns, color='green', alpha=0.7, label='Strategy')
        axs[0, 0].axhline(y=buy_and_hold_total, color='blue', linestyle='--', label='Buy-and-Hold')
        axs[0, 0].set_title('Total Returns by Horizon (%)')
        axs[0, 0].set_xlabel('Horizon')
        axs[0, 0].set_ylabel('Total Return (%)')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Sharpe ratio by horizon
        sharpe_ratios = [backtest_results[h]["sharpe_ratio"] if backtest_results[h]["sharpe_ratio"] is not None else 0 
                         for h in horizons]
        buy_and_hold_sharpe = np.mean(buy_and_hold_returns) / (np.std(buy_and_hold_returns) + 1e-10)
        
        axs[0, 1].bar(horizons, sharpe_ratios, color='purple', alpha=0.7, label='Strategy')
        axs[0, 1].axhline(y=buy_and_hold_sharpe, color='blue', linestyle='--', label='Buy-and-Hold')
        axs[0, 1].set_title('Sharpe Ratio by Horizon')
        axs[0, 1].set_xlabel('Horizon')
        axs[0, 1].set_ylabel('Sharpe Ratio')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Win rate by horizon
        win_rates = [backtest_results[h]["win_rate"] * 100 if backtest_results[h]["win_rate"] is not None else 0 
                     for h in horizons]
        buy_and_hold_win = np.mean(np.array(buy_and_hold_returns) > 0) * 100
        
        axs[1, 0].bar(horizons, win_rates, color='orange', alpha=0.7, label='Strategy')
        axs[1, 0].axhline(y=buy_and_hold_win, color='blue', linestyle='--', label='Buy-and-Hold')
        axs[1, 0].set_title('Win Rate by Horizon (%)')
        axs[1, 0].set_xlabel('Horizon')
        axs[1, 0].set_ylabel('Win Rate (%)')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Equity curves for the best horizon (if available)
        best_horizon = max(horizons, key=lambda h: backtest_results[h]["total_return"] 
                           if backtest_results[h]["total_return"] is not None else -float('inf'))
        
        if backtest_results[best_horizon]["equity_curve"] is not None:
            equity_curve = backtest_results[best_horizon]["equity_curve"]
            buy_hold_equity = np.cumprod(1 + np.array(buy_and_hold_returns))
            
            # Normalize buy_hold_equity to match the length of the equity curve if needed
            if len(buy_hold_equity) > len(equity_curve):
                buy_hold_equity = buy_hold_equity[:len(equity_curve)]
            
            axs[1, 1].plot(equity_curve, color='green', label=f'Strategy (Horizon {best_horizon})')
            axs[1, 1].plot(buy_hold_equity, color='blue', linestyle='--', label='Buy-and-Hold')
            axs[1, 1].set_title('Equity Curve (Best Horizon sample avg)')
            axs[1, 1].set_xlabel('Days')
            axs[1, 1].set_ylabel('Equity ($)')
            axs[1, 1].legend()
            axs[1, 1].grid(True, alpha=0.3)
        else:
            axs[1, 1].text(0.5, 0.5, 'No equity curve available', 
                          horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def _calculate_max_drawdown(self, equity_curve) -> float:
        """
        Calculates the maximum drawdown for a given equity curve.
        
        Args:
            equity_curve: List[float] or np.ndarray: Equity curve values.
            
        Returns:
            float: Maximum drawdown as a decimal (not percentage).
        """
        # Convert to numpy array if it's not already
        equity_array = np.array(equity_curve)
        
        # Check if array is empty or has only one element
        if equity_array.size <= 1:
            return 0.0
            
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        return np.max(drawdown) if drawdown.size > 0 else 0.0