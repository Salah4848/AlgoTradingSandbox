import numpy as np
import copy

class MarketData:
    def __init__(self, data, data_type):
        """
        Basic data class to handle data. Used by the Predictor class. General rule is that it is made of 'samples' and each sample must contain a price array.
        data : is the data stored in the class
        data_type : string to specify type of data that determines the methods defined. Available types : 
          "price_arrays" : 2D numpy arrays or python lists of prices
          "yfinance_Ticker_method_dfs": it is a list (or nparray) of yfinance dataframes (THE FORMAT RETRNED BY .Ticker method, not any other!!!!).
        """
        self.handled_types=["price_arrays","yfinance_Ticker_method_dfs"]
        if data_type not in self.handled_types:
            raise Exception("The data type {data_type} is not handled. Available types: {self.handled_types}")
        self.data_type=data_type #We use this instead of sublasses for simplicity

        if data_type=="price_array":
            self.data=np.array(data) if isinstance(data, list) else data 
        if data_type=="yfinance_Ticker_method_dfs":
            processed = []
            for df in data:
                opens = df['Open'].values
                highs = df['High'].values
                lows = df['Low'].values
                closes = df['Close'].values
                volumes = df['Volume'].values
                dividends = df["Dividends"].values
                stock_splits = df["Stock Splits"].values
                processed.append([opens,highs,lows,closes,volumes,dividends,stock_splits])
            self.data=np.array(processed)
    

    # Inside your class
    def copy(self):
        """
        Returns a deep copy of the current instance.
        """
        return copy.deepcopy(self)
            
    
    def data_slice(self, start_index, end_index):
        """
        Gives slice of data using specfied indexes. Indexes are assumed to be dates. So it returns data from specified start date to end date
        """
        if self.data_type=="price_arrays":
            return MarketData(self.data[:,start_index:end_index],"price_arrays")
        if self.data_type=="yfinance_Ticker_method_dfs":
            out = self.copy()
            out.data = out.data[:, :, start_index:end_index]
            return out
    
    def sample_length(self):
        """
        If its daily price arrays it returns number of days. Gives length of the first sample in our data
        """
        if self.data_type=="price_arrays":
            return self.data[0].shape[0]
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[0,0].shape[0]
    
    def sample_size(self):
        """
        It gives the size of our sample of similar data. For example if our data is arrays of prices of stocks, it returns number of stocks.
        """
        if self.data_type in ["price_arrays","yfinance_Ticker_method_dfs"]:
            return self.data.shape[0]
    
    def get_sample(self, index):
        """
        Return data of the same type but with only one sample specified by the index
        """
        if self.data_type=="price_arrays":
            return MarketData([self.data[index]],"price_arrays")
        if self.data_type=="yfinance_Ticker_method_dfs":
            out = self.copy()
            out.data = np.array([out.data[index]])
            return out
    def get_sample_price(self, index):
        """
        Return the prices array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            return self.data[index]
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 2] # We use closes
        
    def get_sample_open(self, index):
        """
        Return the opens array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No open for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 0]
    def get_sample_high(self, index):
        """
        Return the highs array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No high for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 1]
    def get_sample_low(self, index):
        """
        Return the lows array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No low for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 2]
    def get_sample_close(self, index):
        """
        Return the closes array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No close for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 3]
    def get_sample_volume(self, index):
        """
        Return the volume array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No volume for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 4]
    def get_sample_dividends(self, index):
        """
        Return the dividens array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No dividends for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 5]
    def get_sample_stocksplits(self, index):
        """
        Return the stock splits array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            raise Exception("No stock splits for price_arrays data type")
        if self.data_type=="yfinance_Ticker_method_dfs":
            return self.data[index, 6]