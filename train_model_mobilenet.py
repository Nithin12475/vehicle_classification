import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# -------------------
# Parameters
# -------------------
DATASET_DIR = 'dataset'           # Root folder containing class subfolders
MODEL_PATH = 'model/vehicle_model_mobilenet.h5'
CLASS_INDICES_PATH = 'model/class_indices.json'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4                   # car, bike, bus, truck

# -------------------
# Data Preparation
# -------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_flow = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_flow = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -------------------
# Load Pretrained MobileNetV2
# -------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------
# Compile Model
# -------------------
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------
# Train Model
# -------------------
model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS
)

# -------------------
# Save Model and Class Indices
# -------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(train_flow.class_indices, f)

print(f"Model saved to {MODEL_PATH}")
print(f"Class indices saved to {CLASS_INDICES_PATH}")
