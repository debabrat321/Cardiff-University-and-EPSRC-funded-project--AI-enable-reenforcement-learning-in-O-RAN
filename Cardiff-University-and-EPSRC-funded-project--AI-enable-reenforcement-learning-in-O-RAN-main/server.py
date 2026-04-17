from FederatedPackage import FlaskServerData
from sklearn.linear_model import LinearRegression

server = FlaskServerData.FlaskServer(LinearRegression(),"load", "test_data.csv", 5, 5)
server.StartServer()


