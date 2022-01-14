from os import times

from DataLoader import DataLoader
from StockModel import StockModel


def main():
    tickers = ["KDP", "SBUX","SJM", "QSR"]
    #tickers = ["KDP", "KW", "TGT", "LVS", "AGG", "RY", "JETS", "IBB", "QSR"]
    for t in tickers:
        print(t)
        model = StockModel(t)
        dataLoader = DataLoader()
        data = dataLoader.load(model)
        model.train_model(data)
        prediction = model.predict(data)
        print(prediction)

main()
