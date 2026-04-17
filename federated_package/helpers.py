import torch
import base64
import io

def DeserializeJson(model_state_base64):
    """
    Deserializes a base64 string back into a PyTorch state_dict.
    """
    state_dict_bytes = base64.b64decode(model_state_base64.encode('utf-8'))
    buffer = io.BytesIO(state_dict_bytes)
    state_dict = torch.load(buffer, weights_only=True)
    return state_dict

def SerializeJson(model):
    """
    Serializes a PyTorch model's state_dict into a base64 string.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    state_dict_bytes = buffer.getvalue()
    model_state_base64 = base64.b64encode(state_dict_bytes).decode('utf-8')
    return model_state_base64
