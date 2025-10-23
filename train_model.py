
import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def main(args):
    train_dir = args.dataset
    img_size = (128,128)
    batch_size = args.batch_size
    epochs = args.epochs

    # Data generators (basic augmentation)
    train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_flow = train_gen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_flow = train_gen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_flow.class_indices)
    print("Found classes:", train_flow.class_indices)

    model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(train_flow, validation_data=val_flow, epochs=epochs)

    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model','vehicle_model.h5')
    model.save(model_path)
    print("Saved trained model to", model_path)

    # Save class indices for prediction
    import json
    with open('model/class_indices.json','w') as f:
        json.dump(train_flow.class_indices, f)
    print("Saved class indices to model/class_indices.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a basic CNN for vehicle classification")
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset directory (with subfolders per class)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    main(args)
