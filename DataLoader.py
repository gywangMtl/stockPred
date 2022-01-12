from random import random
from typing import List
from numpy.core.numeric import NaN
from pandas.core.frame import DataFrame
from sklearn import preprocessing
from yahoo_fin import stock_info as si
import numpy as np
import os
import time
import math
from sklearn.model_selection import train_test_split
from collections import deque

import pandas as pd

YEAR_AS_CIRCLE = math.pi*2/366

class DataLoader:
    def __init__(self) -> None:
        pass
    
    def pureLoad(self, model) -> dict:
        ticker = model.ticker
        lookup_step = model.lookup_step
        df = si.get_data(ticker)
        date_now = time.strftime("%Y-%m-%d")
        ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
        df.to_csv(ticker_data_filename)
        result = {}
        result['df'] = df.copy()
        if "date" not in df.columns:
            df["date"] = df.index
        df['future'] = df['adjclose'].shift(-lookup_step)
        return result
  
    def loadCommodityFromCSV(self, dataFrame, commodity) -> list:
        commodityFile = pd.read_csv(os.path.join("data",f"{commodity}.csv"))
        priceList = []
        for date in dataFrame.index:
            dateStr = date.isoformat()[0:10]
            if dateStr in commodityFile.index:
                priceList.append(commodityFile.loc[dateStr, commodity])
            else:
                priceList.append(NaN)
        return priceList 

    def load(self, model, shuffle=True, test_size=0.2):
        ticker = model.ticker
        feature_columns=model.feature_columns
        lookup_step = model.lookup_step
        n_steps = model.sequence_length

        df = si.get_data(ticker)
        result = {}

        result['df'] = df.copy()
        date_now = time.strftime("%Y-%m-%d")
        ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
        df.to_csv(ticker_data_filename)

        if "coffee" in feature_columns:
            df["coffee"] = self.loadCommodityFromCSV(df, "coffee")
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."

        if "date" not in df.columns:
            df["date"] = df.index

        #yearly circle
        df["doy_x"] = df["date"].map(lambda time: math.sin(time.day_of_year * YEAR_AS_CIRCLE))
        df["doy_y"] = df["date"].map(lambda time: math.cos(time.day_of_year * YEAR_AS_CIRCLE))
        feature_columns.append("doy_x")
        feature_columns.append("doy_y")

        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
            np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        result["column_scaler"] = column_scaler

        df['future'] = df['adjclose'].shift(-lookup_step)


        last_sequence = np.array(df[feature_columns].tail(lookup_step))

        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=n_steps)
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])
        last_sequence = list([s[:len(feature_columns)]
                         for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)

        result['last_sequence'] = last_sequence

        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
            test_size=test_size, shuffle=shuffle)

        dates = result["X_test"][:, -1, -1]
        result["test_df"] = result["df"].loc[dates]
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
            keep='first')]
        result["X_train"] = result["X_train"][:, :,
                                          :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :,
                                        :len(feature_columns)].astype(np.float32)
        return result
