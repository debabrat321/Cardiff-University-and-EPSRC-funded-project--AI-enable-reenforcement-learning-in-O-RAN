import pickle
import base64

def DeserializeJson(model):
    modelBase64 = model
    serlizedModel = base64.b64decode(modelBase64.encode('utf-8'))  # Convert back to bytes
    model = pickle.loads(serlizedModel)
    #print('Model has been Deserialized!')
    return model

def SerializeJson(model):
    serlizedModel = pickle.dumps(model)
    modelBase64 = base64.b64encode(serlizedModel).decode('utf-8')  # Convert to base64 string
    #print('Model has been Serialized!')
    return modelBase64
