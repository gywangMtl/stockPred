from os import times
import datetime
from DataLoader import DataLoader
from StockModel import StockModel
import sys

def main(action, tickers):
    today = datetime.date.today()
    if tickers is None:
        return
    if len(tickers)<1:
        return

    for t in tickers:
        print(t)
        model = StockModel(t)
        dataLoader = DataLoader()
        data = dataLoader.load(model)
        if (action == 'train'):
            model.train_model(data)
        else:
            prediction = model.predict(data)
            with open("pred.json", "a") as pred:
                pred.write(f"{t},{today},{prediction}\n")
            print(prediction)

if __name__ == '__main__':
    action = "predict"
    tickers = []
    para_len = len(sys.argv)
    if (para_len >= 2):
        action = sys.argv[1]
    if (para_len >= 3):
        for i in range(2, para_len):
            tickers += [sys.argv[i]]
    
    print(f"doing {action}, stock symbol:[{tickers}]")
    main(action, tickers)
