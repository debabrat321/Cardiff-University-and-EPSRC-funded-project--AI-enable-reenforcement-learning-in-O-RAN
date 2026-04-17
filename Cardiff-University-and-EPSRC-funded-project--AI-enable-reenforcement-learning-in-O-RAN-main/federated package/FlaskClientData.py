#client class
import pandas as pd
import numpy as np
import requests, json
from . import helpers

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
        self.load_and_sort_data()

    def load_and_sort_data(self):
        """
        Load the CSV data and sort it.
        """
        df = pd.read_csv(self.path)
        sorted_df = df.sort_values(by="t")  # Sort the data by the cellIdFeature column
        self.rawClientData = sorted_df



    def LoadClientData(self, cellIdFeature):
        client_data  = self.rawClientData[self.rawClientData[cellIdFeature] == self.clientId]
        subset = len(client_data) // self.maxRounds
        start_index = (self.currentRound - 1) * subset
        end_index = self.currentRound * subset
        self.round_df = client_data.iloc[start_index:end_index]

    
    def PreprocessData(self, dataFrame, target ):
        xTrain = dataFrame.drop(columns=[target])
        yTrain = dataFrame[target]
        return xTrain, yTrain


    def Train(self, xTrain, yTrain):
        self.model.fit(xTrain, yTrain)
        

    def InitialConnection(self, serverUrl, endPoint):
        responseData = (self.session.get(f"{serverUrl}/{endPoint}")).json()
        self.clientId = responseData['clientId']
        self.model = helpers.DeserializeJson(responseData['initalGlobalModel'])
        self.maxRounds = responseData['maxRounds']

    def GetCurrentGlobalModel(self, serverUrl, endPoint):
        responseData = (self.session.get(f"{serverUrl}/{endPoint}")).json()
        self.model = helpers.DeserializeJson(responseData['globalModel'])

    def SendLocalModelToServer(self, serverUrl, endPoint):
        # Prepare the data to be sent to the server
        self.LoadClientData(self.cellIdFeature)
        xTrain, yTrain = self.PreprocessData(self.round_df,self.target)
        self.Train( xTrain, yTrain)
        responseData = {
            "clientId": self.clientId,
            "model": helpers.SerializeJson(self.model),
            "currentRound" : self.currentRound
        }
        headers = {'Content-Type': 'application/json'}
        response = self.session.post(f"{serverUrl}/{endPoint}", data=json.dumps(responseData), headers=headers)
        responseData = response.json()
        self.currentRound += 1



