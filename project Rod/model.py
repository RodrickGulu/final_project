from tensorflow.keras.models import load_model

def load_modell(model_path):
    # Load the Keras model from the specified file path
    model = load_model(model_path)
    return model

# Perform prediction using the loaded model
def predict(model, input_data):
    predictions=model.predict(input_data)
    clsss='Healthy Eye Image' if predictions<0.5 else 'Glaucomatous Eye Image'
    return predictions*100 if predictions>0.5 else 100-(predictions*100), clsss