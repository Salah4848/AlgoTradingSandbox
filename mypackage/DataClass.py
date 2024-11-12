import numpy as np

class MarketData:
    def __init__(self, data, data_type):
        """
        Basic data class to handle data. Used by the Predictor class. General rule is that it is made of 'samples' and each sample must contain a price array.
        data : is the data stored in the class
        data_type : string to specify type of data that determines the methods defined. Available types : 
          "price_arrays" : 2D numpy arrays or python lists of prices
        """
        self.handled_types=["price_arrays"]
        if data_type not in self.handled_types:
            raise Exception("The data type {data_type} is not handled. Available types: {self.handled_types}")
        self.data=np.array(data) if isinstance(data, list) else data 
        self.data_type=data_type #We use this instead of sublasses for simplicity
    
    def data_slice(self, start_index, send_index):
        """
        Gives slice of data using specfied indexes. Indexes are assumed to be dates. So it returns data from specified start date to end date
        """
        if self.data_type=="price_arrays":
            return MarketData(self.data[:,start_index:send_index],"price_arrays")
    
    def sample_size(self):
        """
        It gives the size of our sample of similar data. For example if our data is arrays of prices of stocks, it returns number of stocks.
        """
        if self.data_type=="price_arrays":
            return self.data.shape[0]
    
    def get_sample(self, index):
        """
        Return data of the same type but with only one sample specified by the index
        """
        if self.data_type=="price_arrays":
            return MarketData([self.data[index]],"price_arrays")
    def get_sample_price(self, index):
        """
        Return the prices array (1d np array) of the specified sample
        """
        if self.data_type=="price_arrays":
            return self.data[index]