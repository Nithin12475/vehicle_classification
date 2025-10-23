import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -------------------
# Parameters
# -------------------
MODEL_PATH = 'model/vehicle_model_mobilenet.h5'
CLASS_INDICES_PATH = 'model/class_indices.json'
TARGET_SIZE = (224, 224)  # Must match training size

# -------------------
# Preprocess Image
# -------------------
def preprocess(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0  # Normalize
    x = np.expand_dims(x, axis=0)
    return x

# -------------------
# Main Prediction Function
# -------------------
def main(args):
    # Load model
    model = load_model(MODEL_PATH)
    
    # Load class indices
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse mapping: index -> label
    class_labels = {v: k for k, v in class_indices.items()}
    
    # Preprocess input image
    x = preprocess(args.image)
    
    # Predict
    preds = model.predict(x)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds)
    
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

# -------------------
# CLI Arguments
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict vehicle type from image")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: File '{args.image}' does not exist.")
        exit(1)
    
    main(args)
