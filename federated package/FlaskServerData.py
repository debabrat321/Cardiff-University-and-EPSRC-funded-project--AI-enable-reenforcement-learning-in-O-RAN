import time
from flask import Flask, request, jsonify
from . import helpers
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
from math import sqrt
import pandas as pd



class FlaskServer():
    def __init__(self, globalModel, evalTarget, evalDataPath, maxRounds, maxClients):
        self.globalModel = globalModel
        self.evalTarget = evalTarget
        self.evalDataPath = evalDataPath
        self.clientCounter = 0
        self.roundCounter = 0 
        self.clientsState = {}
        self.maxRounds = maxRounds
        self.maxClients = maxClients
        
        self.app = Flask(__name__)

    def FedAvg(self):
        sumCoefs = None
        sumIntercepts = 0.0
        numOfClients = len(self.clientsState)
        for clientData in self.clientsState.values():
            model = clientData[0]
            if sumCoefs is None:
                sumCoefs = model.coef_.copy()
            else:
                sumCoefs += model.coef_
            sumIntercepts += model.intercept_
        
        avgCoefs = sumCoefs / numOfClients
        avgIntercepts = sumIntercepts / numOfClients

        if hasattr(self.globalModel , 'coef_'):
            prevCoefs = self.globalModel.coef_.copy()
            prevIntercept = self.globalModel.intercept_.copy()
            self.globalModel.coef_ = (prevCoefs + avgCoefs) / 2
            self.globalModel.intercept_ = (prevIntercept + avgIntercepts) / 2
            
        else:
            self.globalModel.coef_ =  avgCoefs
            self.globalModel.intercept_ = avgIntercepts
        

    def CheckAndProcess(self):
        if  len(self.clientsState.keys()) == self.maxClients:
            self.FedAvg()
            print("FED AVG HAS BEEN PERFORMED")
            self.clientsState = {}
            self.roundCounter += 1
            if self.roundCounter == self.maxRounds:
                self.DumpAndEvaluate()


    def DumpAndEvaluate(self):
        # Load the test dataset
        test_data = pd.read_csv(self.evalDataPath)
        print(test_data)
        x_test = test_data.drop(columns=[self.evalTarget])
        print(x_test)
        y_test = test_data[self.evalTarget]
        print(y_test)


        # Predict the values for the test set
        y_pred = self.globalModel.predict(x_test)

        # Calculate the evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mspe = mean_squared_log_error(y_test, y_pred)

        # Print the evaluation metrics
        print(f"R2 Score: {r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Percentage Error: {mspe}")
        print(f"rmse: {rmse}")

        joblib.dump(self.globalModel, 'LRModel.joblib')


    def InitClient(self):
        connectedClientId = self.clientCounter
        responseData = {
            "initalGlobalModel": helpers.SerializeJson(self.globalModel),
            "clientId": connectedClientId,
            "maxRounds" : self.maxRounds
            }
        self.clientCounter += 1
        return jsonify(responseData)

    def UpdateClients(self):
        responseData = {
            "globalModel": helpers.SerializeJson(self.globalModel)
        }
        return jsonify(responseData)

    def ReciveClientUpdates(self):
        requestData = request.get_json()
        self.clientsState[requestData['clientId']] = [helpers.DeserializeJson(requestData['model']), requestData['currentRound']]
        self.CheckAndProcess()
        return jsonify({})

    def StartServer(self):
        @self.app.route('/Init', methods=['GET'])
        def InitRoute():
            return self.InitClient()

        @self.app.route('/UpdateClients', methods=['GET'])
        def UpdateRoute():
            return self.UpdateClients()

        @self.app.route('/ReciveClientUpdates', methods=['POST'])
        def ReciveClientUpdatesRoute():
            return self.ReciveClientUpdates()
            
        self.app.run(host='0.0.0.0', port=5000)
        #eval model 

