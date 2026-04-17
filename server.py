from federated_package import FlaskServerData, models

# Initialize the PyTorch model
# Features: t, cell_id, cat_id, pe_id, has_anomaly (5 features)
model = models.get_model("LinearRegression", input_dim=5)

server = FlaskServerData.FlaskServer(model, "load", "test_data.csv", 5, 5)
server.StartServer()


