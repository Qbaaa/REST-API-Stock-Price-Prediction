from flask import Flask, jsonify, request
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
import pymysql
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import ConfigurationDB as DB

class modelDL:

    def __init__(self):
        self.con = pymysql.connect('localhost', DB.username, DB.password, DB.nameDB)
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

                if "windowSize" in args:
                    self.windowSize = int(args["windowSize"])

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
                result = {'trainData': [], 'predictNextDay': []}

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

                return jsonify(result), 200

            else:
                return "NIE otrzymano ciągu zapytań!!!", 400

        app.run(threaded=False)

    def buildModel(self, model):

        model.add(
            Conv1D(input_shape=(self.windowSize, 1), filters=128, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(
            Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
        model.add(MaxPooling1D(pool_size=1, padding='valid'))
        model.add(Flatten())
        model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
        model.add(Dense(self.predict, activation="relu", kernel_initializer="uniform"))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    def preProcess(self, dateStock, priceStock, minPrice, maxPrice):

        batchX = []
        batchY = []
        batchDateX = []
        batchDateY = []

        normPriceClose = self.normalizationMinMax(priceStock, minPrice, maxPrice)

        for i in range(self.windowSize, len(dateStock) - self.predict + 1):
            for j in range((i - self.windowSize), i):
                batchX.append(normPriceClose[j])
                batchDateX.append(dateStock[j])
            for k in range(self.predict):
                batchY.append(normPriceClose[i + k])
                batchDateY.append(dateStock[i + k])

        batchX = np.reshape(batchX, (int(len(batchX) / self.windowSize), self.windowSize, 1))
        batchY = np.reshape(batchY, (int(len(batchY) / self.predict), self.predict))

        batchDateX = np.reshape(batchDateX, (int(len(batchDateX) / self.windowSize), self.windowSize, 1))
        batchDateY = np.reshape(batchDateY, (int(len(batchDateY) / self.predict), self.predict))

        return batchX, batchY, batchDateX, batchDateY

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
        cursor.execute(DB.sqlSELECT, (symbolStock, '2015-05-31', '2020-05-31'))
        records = cursor.fetchall()
        cursor.close()

        originalDateStock = []
        originalPriceCloseStock = []

        for record in records:
            originalDateStock.append(str(record[0]))
            originalPriceCloseStock.append((record[1]))

        originalMinPriceStock = min(originalPriceCloseStock)
        originalMaxPriceStock = max(originalPriceCloseStock)

        trainX, trainY, trainDateX, trainDateY = self.preProcess(originalDateStock, originalPriceCloseStock, originalMinPriceStock, originalMaxPriceStock)

        history_fit = model.fit(trainX, trainY, validation_split=0.2, batch_size=128, epochs=100, verbose=0)

        predictTrainY = model.predict(trainX)
        predictTrainY = np.reshape(predictTrainY, predictTrainY.shape[0] * self.predict)
        predictTrainY = self.inverseNormalizationMinMax(predictTrainY, originalMinPriceStock, originalMaxPriceStock)

        cursor = self.con.cursor()
        cursor.execute(DB.sqlSELECT2, (symbolStock, dateStart, dateEnd, self.windowSize))
        records = cursor.fetchall()
        cursor.close()

        originalDateStockNexdDay = []
        originalPriceCloseStockNexdDay = []

        for record in records:
            originalDateStockNexdDay.append(str(record[0]))
            originalPriceCloseStockNexdDay.append((record[1]))

        originalDateStockNexdDay.reverse()
        originalPriceCloseStockNexdDay.reverse()

        originalMinPriceStockNextDay = min(originalPriceCloseStockNexdDay)
        originalMaxPriceStockNextDay = max(originalPriceCloseStockNexdDay)

        predicXNextDay = self.normalizationMinMax(originalPriceCloseStockNexdDay, originalMinPriceStockNextDay, originalMaxPriceStockNextDay)
        predicXNextDay = np.reshape(predicXNextDay, (int(len(predicXNextDay) / self.windowSize), self.windowSize, 1))

        predicYNextDay = model.predict(predicXNextDay)
        predicYNextDay = np.reshape(predicYNextDay, predicYNextDay.shape[0] * 1)
        predicYNextDay = self.inverseNormalizationMinMax(predicYNextDay, originalMinPriceStockNextDay, originalMaxPriceStockNextDay)

        loss_values = history_fit.history['loss']
        val_loss_values = history_fit.history['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'b', color = 'blue', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', color='red', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.xticks(epochs)
        plt.rc('font', size=10)
        fig = plt.gcf()
        fig.set_size_inches(18, 7)
        plt.show()

        mae = history_fit.history['mae']
        vmae = history_fit.history['val_mae']
        epochs = range(1, len(mae) + 1)
        plt.plot(epochs, mae, 'b', color='blue', label='Training error')
        plt.plot(epochs, vmae, 'b', color='red', label='Validation error')
        plt.title('Training and validation error')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.xticks(epochs)
        plt.rc('font', size=10)
        fig = plt.gcf()
        fig.set_size_inches(18, 7)
        plt.show()

        ###############################################
        trainY = np.reshape(trainY, trainY.shape[0] * self.predict)
        trainY = self.inverseNormalizationMinMax(trainY, originalMinPriceStock, originalMaxPriceStock)
        trainY = np.reshape(trainY, (int(len(trainY) / self.predict), self.predict))

        predictTrainY = np.reshape(predictTrainY, (int(len(predictTrainY) / self.predict), self.predict))

        print('MAE \t MSE \t MAPE')
        print(mean_absolute_error(trainY, predictTrainY), '\t', mean_squared_error(trainY, predictTrainY), '\t', (np.mean(np.abs((trainY - predictTrainY) / trainY)) * 100))
        predictTrainY = np.reshape(predictTrainY, predictTrainY.shape[0] * self.predict)
        #################################################

        return originalDateStock[self.windowSize:len(originalDateStock)], \
               originalPriceCloseStock[self.windowSize:len(originalPriceCloseStock)], \
               predictTrainY, round(predicYNextDay[0], 2)

modelDL()
