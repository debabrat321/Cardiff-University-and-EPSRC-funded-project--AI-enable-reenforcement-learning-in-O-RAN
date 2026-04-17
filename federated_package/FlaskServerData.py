import time
from flask import Flask, request, jsonify
from . import helpers, models
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.globalModel.to(self.device)
        self.app = Flask(__name__)

    def FedAvg(self):
        """
        Performs Federated Averaging on the received PyTorch state_dicts.
        """
        numOfClients = len(self.clientsState)
        if numOfClients == 0:
            return

        # Get all received state_dicts
        received_states = [clientData[0] for clientData in self.clientsState.values()]
        
        # Initialize average_state with zeros (structure taken from the first received state)
        average_state = {}
        for key in received_states[0].keys():
            average_state[key] = torch.stack([state[key] for state in received_states], 0).mean(0)
        
        # Update the global model
        # Simple averaging: NewGlobal = AverageOfClients
        # Alternatively, you could do: NewGlobal = (PrevGlobal + AverageOfClients) / 2
        # Let's stick to the user's previous logic: (prev + avg) / 2
        
        current_global_state = self.globalModel.state_dict()
        for key in current_global_state.keys():
            current_global_state[key] = (current_global_state[key] + average_state[key]) / 2
            
        self.globalModel.load_state_dict(current_global_state)

    def CheckAndProcess(self):
        if len(self.clientsState.keys()) == self.maxClients:
            self.FedAvg()
            print("FED AVG HAS BEEN PERFORMED")
            self.clientsState = {}
            self.roundCounter += 1
            if self.roundCounter == self.maxRounds:
                self.DumpAndEvaluate()

    def DumpAndEvaluate(self):
        # Load the test dataset
        try:
            test_data = pd.read_csv(self.evalDataPath)
        except FileNotFoundError:
            print(f"Warning: Evaluation data {self.evalDataPath} not found. Skipping evaluation.")
            return

        x_test_np = test_data.drop(columns=[self.evalTarget]).values.astype(np.float32)
        y_test_np = test_data[self.evalTarget].values.astype(np.float32).reshape(-1, 1)

        x_test = torch.from_numpy(x_test_np).to(self.device)
        
        # Predict the values
        self.globalModel.eval()
        with torch.no_grad():
            y_pred_torch = self.globalModel(x_test)
            y_pred = y_pred_torch.cpu().numpy()

        # Calculate metrics (using sklearn as requested/implied)
        r2 = r2_score(y_test_np, y_pred)
        mse = mean_squared_error(y_test_np, y_pred)
        mae = mean_absolute_error(y_test_np, y_pred)
        
        print(f"R2 Score: {r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # Save the model
        torch.save(self.globalModel.state_dict(), 'LRModel_pytorch.pth')
        print("Model saved as LRModel_pytorch.pth")

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
        # requestData['model'] is the base64 string
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

