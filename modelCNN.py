from flask import Flask, jsonify, request
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pymysql
import matplotlib.pyplot as plt
import ConfigurationDB as DB

class modelDL:

    def __init__(self):
        self.con = pymysql.connect('localhost', DB.username, DB.password, DB.nameDB)
        self.windowSize = 5
        self.predict = 1
        app = Flask(__name__)
        self.namefile = ""

        @app.route("/configuration")
        def configurationCNN():

            if request.args:
                args = request.args

                if "structure" in args:
                    structure = args["structure"]
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "conv1" in args:
                    conv1 = int(args["conv1"])
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "conv2" in args:
                    conv2 = int(args["conv2"])
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "dense1" in args:
                    dense1 = int(args["dense1"])
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "initMode" in args:
                    initMode = args["initMode"]
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "activation" in args:
                    activation = args["activation"]
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "optimizer" in args:
                    optimizer = args["optimizer"]
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "batchSize" in args:
                    batchSize = int(args["batchSize"])
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "epoche" in args:
                    epoche = int(args["epoche"])
                else:
                    return "Błędnie otrzymany ciąg zapytań!!!", 400

                if "key" in args:
                    key = args.get("key")

                    if key != DB.keyApi:
                        return "Błąd Uwierzytelniania!!!", 403
                else:
                    return "Wymagane uwierzytlnienie poprzez klucz!!!", 403

                self.windowSize = 5
                if structure == "Input-Conv1D-MaxPooling1D-Flatten-Dense-Dense":
                    model = self.buildModel_1(conv1, dense1, initMode, activation, optimizer)

                    self.namefile = str(structure)+ str(conv1) + str(dense1) + str(initMode) + str(activation) \
                                    + str(optimizer) + str(batchSize) + str(epoche)
                else:
                    model = self.buildModel_2(conv1, conv2, dense1, initMode, activation, optimizer)

                    self.namefile = str(structure)+ str(conv1) + str(conv2) + str(dense1) + str(initMode) + str(activation) \
                                    + str(optimizer) + str(batchSize) + str(epoche)
                result = self.optimizationCNN(model, batchSize, epoche)
                return jsonify(result), 200
            else:
                return "NIE otrzymano ciągu zapytań!!!", 400

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

                model = self.buildModel_2(128, 128, 64, "lecun_uniform", "relu", "adam")
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

    def buildModel_1(self, conv1=128, dense1=64, initMode="uniform", activation="relu", optimizer="adam"):
        model = Sequential()
        model.add(
            Conv1D(input_shape=(self.windowSize, 1),
                   filters=conv1, kernel_size=1, padding='valid', activation=activation, kernel_initializer=initMode))
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(Flatten())
        model.add(Dense(dense1, activation=activation, kernel_initializer=initMode))
        model.add(Dense(self.predict, activation=activation, kernel_initializer=initMode))
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mape'])

        return model

    def buildModel_2(self, conv1=128, conv2=128, dense1=64, initMode="uniform", activation="relu", optimizer="adam"):
        model = Sequential()
        model.add(
            Conv1D(input_shape=(self.windowSize, 1),
                   filters=conv1, kernel_size=1, padding='valid', activation=activation, kernel_initializer=initMode))
        model.add(MaxPooling1D(pool_size=2, padding='valid'))
        model.add(
            Conv1D(filters=conv2, kernel_size=1, padding='valid', activation=activation, kernel_initializer=initMode))
        model.add(MaxPooling1D(pool_size=1, padding='valid'))
        model.add(Flatten())
        model.add(Dense(dense1, activation=activation, kernel_initializer=initMode))
        model.add(Dense(self.predict, activation=activation, kernel_initializer=initMode))
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mape'])

        return model

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
            normPrice.append(0.1 + (((ele - minPrice)*0.9) / (maxPrice - minPrice)))

        return normPrice

    def inverseNormalizationMinMax(self, priceStock, minPrice, maxPrice):

        inverseNormPrice = []

        for ele in priceStock:
            inverseNormPrice.append(((ele - 0.1) * (maxPrice - minPrice)) / 0.9 + minPrice)

        return inverseNormPrice

    def optimizationCNN(self, model, batchSize, epoche):

        cursor = self.con.cursor()
        cursor.execute(DB.sqlSELECT, ('HPQ', '2015-05-31', '2020-05-31'))
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

        valX = trainX[round(0.8*len(trainX)):]
        valY = trainY[round(0.8*len(trainY)):]
        trainX = trainX[:round(0.8*len(trainX))]
        trainY = trainY[:round(0.8*len(trainY))]
        # valDateX = trainDateX[round(0.8*len(trainDateX)):]
        # valDateY = trainDateY[round(0.8*len(trainDateY)):]
        # trainDateX = trainDateX[:round(0.8*len(trainDateX))]
        # trainDateY = trainDateY[:round(0.8*len(trainDateY))]

        history_fit = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=epoche, batch_size=batchSize, verbose=0)

        loss_values = history_fit.history['loss']
        val_loss_values = history_fit.history['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'b', color = 'blue', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', color='red', label='Validation loss')
        plt.rc('font', size=10)
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(epochs)
        fig = plt.gcf()
        fig.set_size_inches(12.8, 7.2)
        # namefile = 'img/opt_LOSS_' + self.namefile + '.png'
        # fig.savefig(namefile, dpi=100)
        plt.show()

        mae = history_fit.history['mape']
        vmae = history_fit.history['val_mape']
        epochs = range(1, len(mae) + 1)
        plt.plot(epochs, mae, 'b', color='blue', label='Training error')
        plt.plot(epochs, vmae, 'b', color='red', label='Validation error')
        plt.rc('font', size=10)
        plt.title('Training and validation error')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE')
        plt.legend()
        plt.xticks(epochs)
        fig = plt.gcf()
        fig.set_size_inches(12.8, 7.2)
        # namefile = 'img/opt_ERROR_' + self.namefile + '.png'
        # fig.savefig(namefile, dpi=100)
        plt.show()

        scoreTrain = model.evaluate(trainX, trainY, verbose=0)
        scoreVal = model.evaluate(valX, valY, verbose=0)
        print(model.metrics_names)
        print("Train: ", scoreTrain)
        print("Val: ", scoreVal)

        # cv = TimeSeriesSplit(n_splits=3)
        # cvScores = []
        # for train_index, test_index in cv.split(valX, valY):
        #     history_fit = model.fit(valX[train_index], valY[train_index], epochs=epoche, batch_size=batchSize, verbose=0)
        #
        #     scores = model.evaluate(valX[test_index], valY[test_index], verbose=0)
        #     print(scores)
        #     print("%s: %.2f%%" % (model.metrics_names[2], scores[2]))
        #     cvScores.append(scores[2])
        #
        # print("%.2f%% (+/- %.2f%%)", (np.mean(cvScores), np.std(cvScores)))

        result = {'Loss': [], 'MAE': [], 'MAPE': []}
        result['Loss'].append({
            'Train': str(scoreTrain[0]),
            'Val': str(scoreVal[0]),
            })
        result['MAE'].append({
            'Train': str(scoreTrain[1]),
            'Val': str(scoreVal[1]),
        })
        result['MAPE'].append({
            'Train': str(scoreTrain[2]),
            'Val': str(scoreVal[2]),
        })

        return result

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
        history_fit = model.fit(trainX, trainY, batch_size=128, epochs=100, verbose=0)

        scoreTrain = model.evaluate(trainX, trainY, verbose=0)
        print(model.metrics_names)
        print("Train: ", scoreTrain)

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

        trainY = np.reshape(trainY, trainY.shape[0] * self.predict)
        trainY = self.inverseNormalizationMinMax(trainY, originalMinPriceStock, originalMaxPriceStock)
        trainY = np.reshape(trainY, (int(len(trainY) / self.predict), self.predict))

        predictTrainY = np.reshape(predictTrainY, (int(len(predictTrainY) / self.predict), self.predict))
        print('MAE \t MSE \t MAPE')
        print(mean_absolute_error(trainY, predictTrainY), '\t', mean_squared_error(trainY, predictTrainY), '\t', (np.mean(np.abs((trainY - predictTrainY) / trainY)) * 100))
        predictTrainY = np.reshape(predictTrainY, predictTrainY.shape[0] * self.predict)

        return originalDateStock[self.windowSize:len(originalDateStock)], \
               originalPriceCloseStock[self.windowSize:len(originalPriceCloseStock)], \
               predictTrainY, round(predicYNextDay[0], 2)

modelDL()