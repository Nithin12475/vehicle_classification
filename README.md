
# Vehicle Image Classification (Basic CNN)

This project contains a simple TensorFlow/Keras-based convolutional neural network (CNN)
to classify vehicle images into classes such as `car`, `bike`, `bus`, `truck`, etc.

## Folder structure

```
vehicle_classification/
├── dataset/            # Put your training images here; subfolders per class
│   ├── car/
│   ├── bike/
│   ├── bus/
│   └── truck/
├── model/              # Trained model and class indices will be saved here
├── test_images/        # Put single images here for quick testing
├── train_model.py
├── predict_image.py
├── requirements.txt
└── README.md
```

## Quick steps (VS Code)

1. Install Python 3.8+ and VS Code.
2. Open the project folder in VS Code.
3. Create a virtual environment:
   - Windows:
     ```
     python -m venv venv
     venv\\Scripts\\activate
     ```
   - macOS / Linux:
     ```
     python -m venv venv
     source venv/bin/activate
     ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Prepare your dataset:
   - Create `dataset/` with subfolders named after each class (e.g., `car`, `bike`, `bus`, `truck`).
   - Place images in the respective folders. Recommended image size is at least 100x100.
6. Train the model:
   ```
   python train_model.py --dataset dataset --epochs 10 --batch_size 16
   ```
   Adjust `--epochs` and `--batch_size` as needed.
7. Predict on a single image:
   ```
   python predict_image.py --image test_images/your_image.jpg
   ```

## Notes & tips
- For better accuracy use transfer learning (e.g., MobileNetV2). This repo intentionally provides a basic CNN for clarity.
- If training is slow on CPU, consider using Google Colab or a machine with a GPU.
- Ensure class folders have balanced numbers of images where possible.
