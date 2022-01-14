import math
import os
import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential

BATCH_SIZE = 64
EPOCHS = 300
COFFE_STOCKS =  ["KDP", "SBUX","SJM", "QSR"]

class StockModel:
    def __init__(self, ticker) -> None:
        self.ticker = ticker
        self.bidirectional=False
        self.optimizer="adam"
        self.cell=LSTM
        self.loss="huber_loss"
        self.n_layers= 3
        self.units=256
        self.dropout=0.4
        self.sequence_length = 50
        self.lookup_step = 3
        self.model_name = self.getModelFileName()
        self.feature_columns = ["adjclose", "volume", "open", "high", "low"]
        if ticker in COFFE_STOCKS:
            print("A coffe stock")
            self.feature_columns.append("coffee")
        self.n_features = len(self.feature_columns) + 2
        self.model=self.create_model()
        # create these folders if they does not exist
        if not os.path.isdir("results"):
            os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir("data"):
            os.mkdir("data")

    def getModelFileName(self) -> str:
        return f"{self.ticker}-{self.loss}-{self.optimizer}-{self.cell.__name__}-seq-{self.sequence_length}-layers-{self.n_layers}-units-{self.units}"
    
    def create_model(self):
        model = Sequential()
        for i in range(self.n_layers):
            if i == 0:
                if self.bidirectional:
                    model.add(Bidirectional(self.cell(self.units, return_sequences=True),
                          batch_input_shape=(None, self.sequence_length, self.n_features)))
                else:
                    model.add(self.cell(self.units, return_sequences=True, batch_input_shape=(
                        None, self.sequence_length, self.n_features)))
            elif i == self.n_layers - 1:
                if self.bidirectional:
                    model.add(Bidirectional(self.cell(self.units, return_sequences=False)))
                else:
                    model.add(self.cell(self.units, return_sequences=False))
            else:
                if self.bidirectional:
                    model.add(Bidirectional(self.cell(self.units, return_sequences=True)))
                else:
                    model.add(self.cell(self.units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(self.dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=self.loss, metrics=["mean_absolute_error"], optimizer=self.optimizer) 
        
        model_path = os.path.join("results", self.model_name) + ".h5"
        if os.path.exists(model_path):
            model.load_weights(model_path)
            datediff = timedelta(days=14)
            lasttime = datetime.fromtimestamp(os.path.getmtime(model_path))
            threshold = (math.tanh((datetime.now() - lasttime)*16 / datediff - 8) + 1) /2.0
            if  random.random() > (1-threshold):
                self.retrain_needed = True
                self.train_epochs = 10
            else:
                self.retrain_needed = False
        else:
            self.retrain_needed = True
            self.train_epochs = EPOCHS

        return model
    
    
    def predict(self, data):
        last_sequence = data["last_sequence"][-self.sequence_length:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = self.model.predict(last_sequence)
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
        current = data["df"]["adjclose"][-1]
        print(current)
        gain = ((predicted_price/current) -1) * 100
        print(gain)
        final_df = self.get_final_df(data)
        self.plot_graph(final_df)
        return predicted_price


    def train_model(self, data):
        if (not self.retrain_needed):
            return
        checkpointer = ModelCheckpoint(os.path.join("results", self.model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))
        self.model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=self.train_epochs,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=0)

    def get_final_df(self, data):
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_pred = self.model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
        test_df = data["test_df"]
        test_df[f"adjclose_{self.lookup_step}"] = y_pred
        test_df[f"true_adjclose_{self.lookup_step}"] = y_test
        test_df.sort_index(inplace=True)
        final_df = test_df
        return final_df

    
    def plot_graph(self, test_df):
        plt.plot(test_df[f'true_adjclose_{self.lookup_step}'], c='b')
        plt.plot(test_df[f'adjclose_{self.lookup_step}'], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])
        plt.savefig(f"graph_{self.ticker}.png", format="png")
        plt.show()
