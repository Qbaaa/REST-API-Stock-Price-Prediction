from flask import Flask, jsonify, request
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import pymysql
from sklearn.metrics import mean_absolute_error

import ConfigurationDB as DB
import keras
import matplotlib.pyplot as plt

class modelDL:
    def __init__(self):
        self.con = pymysql.connect('localhost', DB.username, DB.password, DB.nameDB)
        self.originalDateStock = None
        self.originalPriceCloseStock = None
        self.windowSize = 5
        self.predict = 1
        app = Flask(__name__)

        @app.route("/stock")
        def predictPrice():

            if request.args:
                args = request.args

                if "symbolStock" in args:
                    symbolStock = args["symbolStock"]
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "dateStart" in args:
                    dateStart = args.get("dateStart")
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "dateEnd" in args:
                    dateEnd = args.get("dateEnd")
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "key" in args:
                    key = args.get("key")

                    if key != DB.keyApi:
                        return "Błąd Uwierzytelniania!!!", 403

                else:
                    return "Wymagane uwierzytlnienie poprzez klucz!!!", 403

                model = Sequential()
                self.buildModel(model)

                date, originalPrice, predictPricePast, predictNextDay = self.trainModel(model, symbolStock, dateStart, dateEnd)
                result = {}
                result['trainData'] = []
                result['predictNextDay'] = []

                for index in range(len(date)):
                    result['trainData'].append({
                                                'date': str(date[index]),
                                                'originalPrice': originalPrice[index],
                                                'predictPrice': predictPricePast[index]
                                                })

                result['predictNextDay'].append({
                                                'date': dateEnd,
                                                'price': predictNextDay
                                                })

                print(dateEnd)
                print(predictNextDay)

                return jsonify(result), 200

            else:
                return "NIE otrzymano ciągu zapytań!!!", 400

        app.run(threaded=False)

    def buildModel(self, model):

        model.add(Dense(128, input_shape=(self.windowSize, 1)))
        model.add(
            Conv1D(filters=112, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(
            Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
        model.add(MaxPooling1D(pool_size=1, padding='valid'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(100, activation="relu", kernel_initializer="uniform"))
        model.add(Dense(self.predict, activation="relu", kernel_initializer="uniform"))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    def preProcess(self, dateStock, priceStock):

        self.trainX = []
        self.trainY = []

        normPriceClose = self.normalizationMinMax(priceStock, self.originalMinPriceStock, self.originalMaxPriceStock)

        for i in range(self.windowSize, len(dateStock) - self.predict + 1):
            for j in range((i - self.windowSize), i):
                self.trainX.append(normPriceClose[j])
            for k in range(self.predict):
                self.trainY.append(normPriceClose[i + k])

        self.trainX = np.reshape(self.trainX, (int(len(self.trainX) / self.windowSize), self.windowSize, 1))
        self.trainY = np.reshape(self.trainY, (int(len(self.trainY) / self.predict), self.predict))

    def normalizationMinMax(self, priceStock, minPrice, maxPrice):

        normPrice = []

        for ele in priceStock:
            normPrice.append((ele - minPrice) / (maxPrice - minPrice))

        return normPrice

    def inverseNormalizationMinMax(self, priceStock, minPrice, maxPrice):

        inverseNormPrice = []

        for ele in priceStock:
            inverseNormPrice.append(ele * (maxPrice - minPrice) + minPrice)

        return inverseNormPrice

    def trainModel(self, model, symbolStock, dateStart, dateEnd):

        cursor = self.con.cursor()
        cursor.execute("SELECT date, price_close FROM companies WHERE symbol = %s AND date >= %s AND date <= %s", (symbolStock, dateStart, dateEnd))
        records = cursor.fetchall()
        cursor.close()

        self.originalDateStock = []
        self.originalPriceCloseStock = []

        for record in records:
            self.originalDateStock.append(str(record[0]))
            self.originalPriceCloseStock.append((record[1]))

        self.originalMinPriceStock = min(self.originalPriceCloseStock)
        self.originalMaxPriceStock = max(self.originalPriceCloseStock)

        self.preProcess(self.originalDateStock, self.originalPriceCloseStock)

        history_fit = model.fit(self.trainX, self.trainY, validation_split=0.2, batch_size=128, epochs=35, verbose=2)

        predictTrainY = model.predict(self.trainX)
        predictTrainY = np.reshape(predictTrainY, predictTrainY.shape[0] * self.predict)
        predictTrainY = self.inverseNormalizationMinMax(predictTrainY, self.originalMinPriceStock, self.originalMaxPriceStock)

        predicXNextDay = self.originalPriceCloseStock[-5:]
        predicXNextDay = self.normalizationMinMax(predicXNextDay, self.originalMinPriceStock, self.originalMaxPriceStock)
        predicXNextDay = np.reshape(predicXNextDay, (int(len(predicXNextDay) / self.windowSize), self.windowSize, 1))

        predicYNextDay = model.predict(predicXNextDay)
        predicYNextDay = np.reshape(predicYNextDay, predicYNextDay.shape[0] * 1)
        predicYNextDay = self.inverseNormalizationMinMax(predicYNextDay, self.originalMinPriceStock, self.originalMaxPriceStock)

        loss_values = history_fit.history['loss']
        val_loss_values = history_fit.history['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'b', color = 'blue', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', color='red', label='Validation loss')
        plt.rc('font', size=18)
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(epochs)
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.show()

        mae = history_fit.history['mae']
        vmae = history_fit.history['val_mae']
        epochs = range(1, len(mae) + 1)
        plt.plot(epochs, mae, 'b', color='blue', label='Training error')
        plt.plot(epochs, vmae, 'b', color='red', label='Validation error')
        plt.title('Training and validation error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.xticks(epochs)
        fig = plt.gcf()
        fig.set_size_inches(15, 7)
        plt.show()

        ###############################################
        print(self.trainY.shape)

        self.trainY = np.reshape(self.trainY, self.trainY.shape[0]*self.predict)
        self.trainY = self.inverseNormalizationMinMax(self.trainY, self.originalMinPriceStock, self.originalMaxPriceStock)
        self.trainY = np.reshape(self.trainY, (int(len(self.trainY) / self.predict), self.predict))

        predictTrainY = np.reshape(predictTrainY, (int(len(predictTrainY) / self.predict), self.predict))

        print(self.originalPriceCloseStock[0])
        print(self.trainX[0])

        

        print(predictTrainY[0])

        print('mean absolute error \t mean absolute percentage error')
        print((mean_absolute_error(self.trainY, predictTrainY)), '\t', (np.mean(np.abs((self.trainY - predictTrainY) / self.trainY)) * 100))
        predictTrainY = np.reshape(predictTrainY, predictTrainY.shape[0] * self.predict)
        #################################################

        return self.originalDateStock[5:len(self.originalDateStock)], self.originalPriceCloseStock[5:len(self.originalPriceCloseStock)], predictTrainY, predicYNextDay[0]

modelDL()
