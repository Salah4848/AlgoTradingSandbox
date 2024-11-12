import numpy as np
import pandas as pd


def calculate_technical_indicators(highs, lows, closes, volumes):
    # Initialize a dictionary to store the indicators
    indicators_dict = {}
    
    # Calculate each indicator
    sma_vals = sma(closes, window=20)
    ema_vals = ema(closes, window=20)
    rsi_vals = rsi(closes, window=14)
    macd_line, signal_line = macd(closes)
    upper_band, lower_band = bollinger_bands(closes)
    atr_vals = atr(highs, lows, closes, window=14)
    vwap_vals = vwap(closes, volumes)
    cci_vals = cci(highs, lows, closes, window=20)
    stochastic_vals = stochastic_oscillator(closes, window=14)
    adx_vals = adx(highs, lows, closes, window=14)
    mfi_vals = mfi(highs, lows, closes, volumes, window=14)
    williams_r_vals = williams_r(closes, window=14)
    tsi_vals = tsi(closes, r=25, s=13)
    chaikin_vals = chaikin_oscillator(highs, lows, closes, volumes, short_window=3, long_window=10)
    conv_line, base_line, lead_span_a, lead_span_b = ichimoku_cloud(closes)
    keltner_upper, keltner_lower = keltner_channel(closes)
    parabolic_sar_vals = parabolic_sar(highs, lows)
    obv_vals = obv(closes, volumes)
    ad_line_vals = ad_line(highs, lows, closes, volumes)
    aroon_oscillator_vals = aroon_oscillator(closes, window=25)

    # Length of the stock data
    data_length = len(closes)
    
    # Add each indicator to the dictionary, padding with NaNs at the beginning
    indicators_dict['SMA'] = np.concatenate([np.full(data_length - len(sma_vals), np.nan), sma_vals])
    indicators_dict['EMA'] = np.concatenate([np.full(data_length - len(ema_vals), np.nan), ema_vals])
    indicators_dict['RSI'] = np.concatenate([np.full(data_length - len(rsi_vals), np.nan), rsi_vals])
    indicators_dict['MACD'] = np.concatenate([np.full(data_length - len(macd_line), np.nan), macd_line])
    indicators_dict['MACD_Signal'] = np.concatenate([np.full(data_length - len(signal_line), np.nan), signal_line])
    indicators_dict['Bollinger_Upper'] = np.concatenate([np.full(data_length - len(upper_band), np.nan), upper_band])
    indicators_dict['Bollinger_Lower'] = np.concatenate([np.full(data_length - len(lower_band), np.nan), lower_band])
    indicators_dict['ATR'] = np.concatenate([np.full(data_length - len(atr_vals), np.nan), atr_vals])
    indicators_dict['VWAP'] = np.concatenate([np.full(data_length - len(vwap_vals), np.nan), vwap_vals])
    indicators_dict['CCI'] = np.concatenate([np.full(data_length - len(cci_vals), np.nan), cci_vals])
    indicators_dict['Stochastic_Oscillator'] = np.concatenate([np.full(data_length - len(stochastic_vals), np.nan), stochastic_vals])
    indicators_dict['ADX'] = np.concatenate([np.full(data_length - len(adx_vals), np.nan), adx_vals])
    indicators_dict['MFI'] = np.concatenate([np.full(data_length - len(mfi_vals), np.nan), mfi_vals])
    indicators_dict['Williams_R'] = np.concatenate([np.full(data_length - len(williams_r_vals), np.nan), williams_r_vals])
    indicators_dict['TSI'] = np.concatenate([np.full(data_length - len(tsi_vals), np.nan), tsi_vals])
    indicators_dict['Chaikin_Oscillator'] = np.concatenate([np.full(data_length - len(chaikin_vals), np.nan), chaikin_vals])
    indicators_dict['Ichimoku_Conversion_Line'] = np.concatenate([np.full(data_length - len(conv_line), np.nan), conv_line])
    indicators_dict['Ichimoku_Base_Line'] = np.concatenate([np.full(data_length - len(base_line), np.nan), base_line])
    #indicators_dict['Ichimoku_Leading_Span_A'] = np.concatenate([np.full(data_length - len(lead_span_a), np.nan), lead_span_a])
    #indicators_dict['Ichimoku_Leading_Span_B'] = np.concatenate([np.full(data_length - len(lead_span_b), np.nan), lead_span_b])
    indicators_dict['Keltner_Upper'] = np.concatenate([np.full(data_length - len(keltner_upper), np.nan), keltner_upper])
    indicators_dict['Keltner_Lower'] = np.concatenate([np.full(data_length - len(keltner_lower), np.nan), keltner_lower])
    indicators_dict['Parabolic_SAR'] = np.concatenate([np.full(data_length - len(parabolic_sar_vals), np.nan), parabolic_sar_vals])
    indicators_dict['OBV'] = np.concatenate([np.full(data_length - len(obv_vals), np.nan), obv_vals])
    indicators_dict['AD_Line'] = np.concatenate([np.full(data_length - len(ad_line_vals), np.nan), ad_line_vals])
    indicators_dict['Aroon_Oscillator'] = np.concatenate([np.full(data_length - len(aroon_oscillator_vals), np.nan), aroon_oscillator_vals])

    # Create a DataFrame from the dictionary
    indicators_df = pd.DataFrame(indicators_dict)
    
    return indicators_df

def sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def ema(prices, window):
    alpha = 2 / (window + 1)
    ema_values = np.zeros_like(prices)
    ema_values[0] = prices[0]
    for t in range(1, len(prices)):
        ema_values[t] = alpha * prices[t] + (1 - alpha) * ema_values[t - 1]
    return ema_values

def rsi(prices, window=14):
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gains, np.ones(window) / window, mode='valid')
    avg_loss = np.convolve(losses, np.ones(window) / window, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def macd(prices, fast_window=12, slow_window=26, signal_window=9):
    fast_ema = ema(prices, fast_window)
    slow_ema = ema(prices, slow_window)
    macd_line = fast_ema[slow_window-1:] - slow_ema[slow_window-1:]
    signal_line = ema(macd_line, signal_window)
    return macd_line, signal_line

def bollinger_bands(prices, window=20, num_std=2):
    sma_values = sma(prices, window)
    rolling_std = np.array([np.std(prices[i:i + window]) for i in range(len(prices) - window + 1)])
    upper_band = sma_values + num_std * rolling_std
    lower_band = sma_values - num_std * rolling_std
    return upper_band, lower_band

def atr(highs, lows, closes, window=14):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.convolve(tr, np.ones(window) / window, mode='valid')

def vwap(prices, volumes):
    cumulative_price_volume = np.cumsum(prices * volumes)
    cumulative_volume = np.cumsum(volumes)
    return cumulative_price_volume / cumulative_volume

def cci(highs, lows, closes, window=20):
    typical_price = (highs + lows + closes) / 3
    sma_values = sma(typical_price, window)
    mean_deviation = np.array([np.mean(np.abs(typical_price[i:i + window] - sma_values[i])) for i in range(len(sma_values))])
    return (typical_price[window - 1:] - sma_values) / (0.015 * mean_deviation)

def stochastic_oscillator(prices, window=14):
    highest_highs = np.array([np.max(prices[i:i + window]) for i in range(len(prices) - window + 1)])
    lowest_lows = np.array([np.min(prices[i:i + window]) for i in range(len(prices) - window + 1)])
    close = prices[window - 1:]
    return (close - lowest_lows) / (highest_highs - lowest_lows) * 100

def adx(highs, lows, closes, window=14):
    plus_dm = np.where(highs[1:] > highs[:-1], highs[1:] - highs[:-1], 0)
    minus_dm = np.where(lows[:-1] > lows[1:], lows[:-1] - lows[1:], 0)
    atr_values = atr(highs, lows, closes, window)
    plus_di = 100 * np.convolve(plus_dm, np.ones(window) / window, mode='valid') / (atr_values + 1e-10)
    minus_di = 100 * np.convolve(minus_dm, np.ones(window) / window, mode='valid') / (atr_values + 1e-10)
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    return np.convolve(dx, np.ones(window) / window, mode='valid')

def mfi(highs, lows, closes, volumes, window=14):
    typical_price = (highs + lows + closes) / 3
    money_flow = typical_price * volumes
    positive_flow = np.where(typical_price[1:] > typical_price[:-1], money_flow[1:], 0)
    negative_flow = np.where(typical_price[1:] < typical_price[:-1], money_flow[1:], 0)
    pos_mf = np.convolve(positive_flow, np.ones(window) / window, mode='valid')
    neg_mf = np.convolve(negative_flow, np.ones(window) / window, mode='valid')
    mfi_values = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    return mfi_values

def williams_r(prices, window=14):
    highest_highs = np.array([np.max(prices[i:i + window]) for i in range(len(prices) - window + 1)])
    lowest_lows = np.array([np.min(prices[i:i + window]) for i in range(len(prices) - window + 1)])
    close = prices[window - 1:]
    return ((highest_highs - close) / (highest_highs - lowest_lows + 1e-10)) * -100

def tsi(prices, r=25, s=13):
    momentum = np.diff(prices)
    abs_momentum = np.abs(momentum)
    def double_ema(values, window):
        ema1 = np.convolve(values, np.ones(window) / window, mode='valid')
        return np.convolve(ema1, np.ones(window) / window, mode='valid')
    double_smoothed_momentum = double_ema(momentum, r)
    double_smoothed_abs_momentum = double_ema(abs_momentum, r)
    return 100 * (double_smoothed_momentum / (double_smoothed_abs_momentum + 1e-10))

def chaikin_oscillator(highs, lows, closes, volumes, short_window=3, long_window=10):
    adl = np.cumsum(((closes - lows) - (highs - closes)) / (highs - lows + 1e-10) * volumes)
    short_ema = np.convolve(adl, np.ones(short_window) / short_window, mode='valid')
    long_ema = np.convolve(adl, np.ones(long_window) / long_window, mode='valid')
    return short_ema[-len(long_ema):] - long_ema

def ichimoku_cloud(prices, window1=9, window2=26, window3=52):
    # Ensure there is enough data
    if len(prices) < max(window1, window2, window3):
        raise ValueError("Not enough price data for Ichimoku Cloud calculation")
    
    # Initialize arrays filled with NaN
    conv_line = np.full(len(prices), np.nan)
    base_line = np.full(len(prices), np.nan)
    lead_span_a = np.full(len(prices) + window2, np.nan)  # Offset Lead Span A
    lead_span_b = np.full(len(prices) + window2, np.nan)  # Offset Lead Span B
    
    # Calculate Conversion Line (Tenkan-sen)
    for i in range(window1 - 1, len(prices)):
        conv_line[i] = (np.max(prices[i - window1 + 1:i + 1]) + np.min(prices[i - window1 + 1:i + 1])) / 2
    
    # Calculate Base Line (Kijun-sen)
    for i in range(window2 - 1, len(prices)):
        base_line[i] = (np.max(prices[i - window2 + 1:i + 1]) + np.min(prices[i - window2 + 1:i + 1])) / 2
    
    # Calculate Leading Span A (Senkou Span A), offset by 26 periods forward
    for i in range(window2 - 1, len(prices)):
        if not np.isnan(conv_line[i]) and not np.isnan(base_line[i]):
            lead_span_a[i + window2] = (conv_line[i] + base_line[i]) / 2
    
    # Calculate Leading Span B (Senkou Span B), offset by 26 periods forward
    for i in range(window3 - 1, len(prices)):
        lead_span_b[i + window2] = (np.max(prices[i - window3 + 1:i + 1]) + np.min(prices[i - window3 + 1:i + 1])) / 2
    
    # Trim the extra length added by the forward projection for consistency
    lead_span_a = lead_span_a[:len(prices)]
    lead_span_b = lead_span_b[:len(prices)]
    
    return conv_line, base_line, lead_span_a, lead_span_b
    
    return conv_line, base_line, lead_span_a, lead_span_b

def keltner_channel(prices, window=20, atr_window=14, multiplier=1.5):
    # Calculate the Exponential Moving Average (EMA) of the price
    ema_center = ema(prices, window)  # Assuming you already have an 'ema' function defined.
    
    # Calculate the True Range (TR) and Average True Range (ATR)
    tr = np.maximum(prices[1:] - prices[:-1], np.abs(prices[1:] - prices[:-1]))
    atr = np.concatenate([np.full(atr_window - 1, np.nan), np.convolve(tr, np.ones(atr_window) / atr_window, mode='valid')])
    
    # Calculate rolling mean and rolling squared mean using mode='same' to align lengths
    rolling_mean = np.convolve(prices, np.ones(window) / window, mode='same')
    rolling_squared_mean = np.convolve(prices**2, np.ones(window) / window, mode='same')
    
    # Calculate rolling standard deviation
    rolling_std = np.sqrt(rolling_squared_mean - rolling_mean**2)
    
    # Calculate Keltner Channel upper and lower bands
    upper_band = ema_center + multiplier * rolling_std
    lower_band = ema_center - multiplier * rolling_std
    
    # Ensure the length of both upper_band and lower_band match the original prices array
    # You may need to handle the NaN values at the beginning based on the window size
    upper_band = np.concatenate([np.full(window-1, np.nan), upper_band[window-1:]])
    lower_band = np.concatenate([np.full(window-1, np.nan), lower_band[window-1:]])
    
    return upper_band, lower_band

def parabolic_sar(highs, lows, acceleration_factor=0.02, max_af=0.2):
    sar = np.zeros_like(highs)
    trend_up = True
    ep = lows[0] if trend_up else highs[0]
    af = acceleration_factor
    for i in range(1, len(highs)):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if trend_up:
            if lows[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                af = acceleration_factor
                ep = highs[i]
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + acceleration_factor, max_af)
        else:
            if highs[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                af = acceleration_factor
                ep = lows[i]
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + acceleration_factor, max_af)
    return sar

def obv(prices, volumes):
    delta = np.diff(prices)
    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    obv_values = np.cumsum(direction * volumes[1:])
    return np.insert(obv_values, 0, 0)  # Start with 0 at the beginning

def ad_line(highs, lows, closes, volumes):
    """
    Accumulation/Distribution (A/D) Line
    highs, lows, closes: 1D numpy arrays of high, low, and close prices
    volumes: 1D numpy array of volumes
    """
    ad_values = np.zeros(len(closes))
    for i in range(len(closes)):
        clv = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i]) if (highs[i] != lows[i]) else 0
        ad_values[i] = ad_values[i - 1] + (clv * volumes[i]) if i > 0 else clv * volumes[i]
    return ad_values

def aroon_oscillator(prices, window=25):
    """
    Aroon Oscillator
    prices: 1D numpy array of prices
    window: Number of periods for high/low search (default 25)
    """
    aroon_up = np.array([(window - np.argmax(prices[i:i+window])) * 100 / window for i in range(len(prices) - window + 1)])
    aroon_down = np.array([(window - np.argmin(prices[i:i+window])) * 100 / window for i in range(len(prices) - window + 1)])
    return aroon_up - aroon_down
