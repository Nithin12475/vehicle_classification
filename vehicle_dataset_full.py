# vehicle_dataset_full.py

from icrawler.builtin import BingImageCrawler
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# ---------------------------
# Step 1: Configuration
# ---------------------------
classes = ["car", "motorbike", "truck", "bus"]
num_images_per_class = 500  # number of images to download per class
augmentations_per_image = 5  # number of augmented images per original
download_root = "dataset_raw"
augment_root = "dataset_final"

os.makedirs(download_root, exist_ok=True)
os.makedirs(augment_root, exist_ok=True)

# ---------------------------
# Step 2: Download Images
# ---------------------------
print("Downloading images from Bing...")
for cls in classes:
    cls_dir = os.path.join(download_root, cls)
    os.makedirs(cls_dir, exist_ok=True)
    crawler = BingImageCrawler(storage={'root_dir': cls_dir})
    crawler.crawl(keyword=cls, max_num=num_images_per_class)
print("Download complete!")

# ---------------------------
# Step 3: Augment Images
# ---------------------------
print("Starting augmentation...")
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for cls in classes:
    input_dir = os.path.join(download_root, cls)
    output_dir = os.path.join(augment_root, cls)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        try:
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir,
                                      save_prefix=cls, save_format='jpg'):
                i += 1
                if i >= augmentations_per_image:
                    break
        except Exception as e:
            print(f"Skipping {filename}: {e}")

print(f"Augmentation complete! Final dataset is in '{augment_root}'")

