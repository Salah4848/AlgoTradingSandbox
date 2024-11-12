from mypackage import *

def main():
    data = np.array([i for i in range(1,101)])
    print(data)
    maPred = BasicLSTM(MarketData([data],"price_arrays"),"models/basic_lstm_stock_model.pt")
    results = maPred.benchmark()
    print(results)
if __name__=="__main__":
    main()
 