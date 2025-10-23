from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# Classes of vehicles
classes = ["car", "bike", "truck", "bus"]

# Input folder (original images)
input_root = "dataset"

# Output folder (augmented images)
output_root = "dataset_augmented"
os.makedirs(output_root, exist_ok=True)

# Image augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,       # rotate images up to 30 degrees
    width_shift_range=0.2,   # horizontal shift
    height_shift_range=0.2,  # vertical shift
    shear_range=0.15,        # shear transformation
    zoom_range=0.2,          # zoom in/out
    horizontal_flip=True,    # flip horizontally
    fill_mode='nearest'      # fill empty pixels
)

# Loop through each class
for cls in classes:
    input_dir = os.path.join(input_root, cls)
    output_dir = os.path.join(output_root, cls)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        try:
            # Load image
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate 5 augmented images per original image
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=cls, save_format='jpg'):
                i += 1
                if i >= 5:
                    break
        except Exception as e:
            print(f"Skipping {filename}: {e}")

print("Augmentation complete! Augmented dataset is in 'dataset_augmented/'")
