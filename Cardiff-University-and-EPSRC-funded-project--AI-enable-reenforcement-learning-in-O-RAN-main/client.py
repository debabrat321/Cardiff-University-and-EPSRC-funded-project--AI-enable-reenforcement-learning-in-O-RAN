from FederatedPackage import FlaskClientData
import time

serverUrl = "http://127.0.0.1:5000" 

client = FlaskClientData.FlaskClient("train_data.csv", "cell_id", "load")

client.InitialConnection(serverUrl, "Init")
client.SendLocalModelToServer(serverUrl, "ReciveClientUpdates")
time.sleep(10)

while client.currentRound < client.maxRounds + 1:
    client.GetCurrentGlobalModel(serverUrl, "UpdateClients")
    client.SendLocalModelToServer(serverUrl, "ReciveClientUpdates")
    time.sleep(10)









