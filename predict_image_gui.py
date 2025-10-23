import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "model/vehicle_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"
TARGET_SIZE = (128, 128)  # same as training

# Load the model and class labels
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

def preprocess(img_path, target_size=TARGET_SIZE):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_path):
    x = preprocess(img_path)
    preds = model.predict(x)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_id]
    return class_labels[class_id], confidence

# Tkinter GUI
def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select Vehicle Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        try:
            predicted_class, confidence = predict(file_path)
            messagebox.showinfo(
                "Prediction",
                f"Predicted class: {predicted_class}\nConfidence: {confidence:.2f}"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

# GUI Window
root = tk.Tk()
root.title("Vehicle Image Classifier")
root.geometry("300x100")

btn = tk.Button(root, text="Select Image", command=browse_image, width=25, height=2)
btn.pack(pady=20)

root.mainloop()
