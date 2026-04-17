#client class
import pandas as pd
import numpy as np
import requests, json
import torch
import torch.nn as nn
import torch.optim as optim
from . import helpers, models

class FlaskClient:
    def __init__(self, path, cellIdFeature, target):
        self.clientId = None
        self.model = None
        self.session = requests.Session()
        self.currentRound = 1
        self.round_df = None
        self.rawClientData = None
        self.path = path
        self.cellIdFeature = cellIdFeature
        self.maxRounds = None
        self.target = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_and_sort_data()

    def load_and_sort_data(self):
        """
        Load the CSV data and sort it.
        """
        try:
            df = pd.read_csv(self.path)
            sorted_df = df.sort_values(by="t")  # Sort the data by the cellIdFeature column
            self.rawClientData = sorted_df
        except FileNotFoundError:
            print(f"Warning: Data file {self.path} not found. Creating dummy data for testing.")
            # Dummy data structure matching test_data.csv: t,cell_id,cat_id,pe_id,load,has_anomaly
            data = {
                't': np.arange(100),
                'cell_id': np.random.randint(0, 5, 100),
                'cat_id': np.random.randint(0, 3, 100),
                'pe_id': np.random.randint(0, 3, 100),
                'load': np.random.rand(100) * 100,
                'has_anomaly': np.random.randint(0, 2, 100)
            }
            self.rawClientData = pd.DataFrame(data)

    def LoadClientData(self, cellIdFeature):
        client_data = self.rawClientData[self.rawClientData[cellIdFeature] == self.clientId]
        if len(client_data) == 0:
            # If no data for this client, use a sample of the raw data to avoid division by zero
            client_data = self.rawClientData.sample(min(10, len(self.rawClientData)))
            
        subset = max(1, len(client_data) // self.maxRounds)
        start_index = (self.currentRound - 1) * subset
        end_index = self.currentRound * subset
        self.round_df = client_data.iloc[start_index:end_index]

    def PreprocessData(self, dataFrame, target):
        x = dataFrame.drop(columns=[target]).values.astype(np.float32)
        y = dataFrame[target].values.astype(np.float32).reshape(-1, 1)
        
        xTrain = torch.from_numpy(x).to(self.device)
        yTrain = torch.from_numpy(y).to(self.device)
        return xTrain, yTrain

    def Train(self, xTrain, yTrain, epochs=10, lr=0.01):
        if self.model is None:
            input_dim = xTrain.shape[1]
            self.model = models.get_model("LinearRegression", input_dim).to(self.device)
        
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(xTrain)
            loss = criterion(outputs, yTrain)
            loss.backward()
            optimizer.step()
        
        print(f"Round {self.currentRound} training complete. Loss: {loss.item():.4f}")

    def InitialConnection(self, serverUrl, endPoint):
        responseData = (self.session.get(f"{serverUrl}/{endPoint}")).json()
        self.clientId = responseData['clientId']
        self.maxRounds = responseData['maxRounds']
        
        # We need data to know the input dimension for model initialization
        self.LoadClientData(self.cellIdFeature)
        xTrain, _ = self.PreprocessData(self.round_df, self.target)
        input_dim = xTrain.shape[1]
        
        self.model = models.get_model("LinearRegression", input_dim).to(self.device)
        self.model.load_state_dict(helpers.DeserializeJson(responseData['initalGlobalModel']))

    def GetCurrentGlobalModel(self, serverUrl, endPoint):
        responseData = (self.session.get(f"{serverUrl}/{endPoint}")).json()
        self.model.load_state_dict(helpers.DeserializeJson(responseData['globalModel']))

    def SendLocalModelToServer(self, serverUrl, endPoint):
        # Prepare the data to be sent to the server
        self.LoadClientData(self.cellIdFeature)
        xTrain, yTrain = self.PreprocessData(self.round_df, self.target)
        self.Train(xTrain, yTrain)
        
        responseData = {
            "clientId": self.clientId,
            "model": helpers.SerializeJson(self.model),
            "currentRound": self.currentRound
        }
        headers = {'Content-Type': 'application/json'}
        response = self.session.post(f"{serverUrl}/{endPoint}", data=json.dumps(responseData), headers=headers)
        responseData = response.json()
        self.currentRound += 1



