
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

def load_class_indices(path='model/class_indices.json'):
    with open(path,'r') as f:
        class_indices = json.load(f)
    # invert mapping
    inv = {v:k for k,v in class_indices.items()}
    return inv

def preprocess(img_path, target_size=(128,128)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def main(args):
    model_path = args.model
    img_path = args.image

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first by running train_model.py")

    model = load_model(model_path)
    class_map = load_class_indices(os.path.join(os.path.dirname(model_path),'class_indices.json'))

    x = preprocess(img_path)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = class_map[idx]
    confidence = float(preds[idx])

    print(f"Prediction: {label} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict vehicle class for an image")
    parser.add_argument('--model', type=str, default='model/vehicle_model.h5', help='Path to trained model .h5')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to classify')
    args = parser.parse_args()
    main(args)
