import torch
import torch.nn as nn

class TorchLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(TorchLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def get_model(model_name, input_dim, output_dim=1):
    if model_name.lower() == "linearregression":
        return TorchLinearRegression(input_dim, output_dim)
    # Future expansions like LSTM or GRU can go here
    raise ValueError(f"Model {model_name} not supported")
