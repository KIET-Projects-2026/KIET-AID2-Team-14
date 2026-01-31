import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -----------------------------
# 1️⃣ Set dataset paths
# -----------------------------
train_dir = r"C:\Users\nawyah11112004\Downloads\plant-disease-main (1)\plant-disease-main\dataset\train"
val_dir   = r"C:\Users\nawyah11112004\Downloads\plant-disease-main (1)\plant-disease-main\dataset\val"

# -----------------------------
# 2️⃣ Check if paths exist
# -----------------------------
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training folder not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation folder not found: {val_dir}")

print("[INFO] Dataset folders verified!")

# -----------------------------
# 3️⃣ Remove corrupted images
# -----------------------------
def remove_bad_images(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()
            except:
                print("[WARNING] Removing bad image:", path)
                os.remove(path)

print("[INFO] Checking & cleaning images...")
remove_bad_images(train_dir)
remove_bad_images(val_dir)
print("[INFO] Image cleaning complete!")

# -----------------------------
# 4️⃣ Image Generators (FAST)
# -----------------------------
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print("[INFO] Data loaded successfully!")
print("[INFO] Number of classes:", len(train_generator.class_indices))

# -----------------------------
# 5️⃣ Build Simple CNN (FAST)
# -----------------------------
num_classes = len(train_generator.class_indices)

model = models.Sequential([
    layers.Input((128, 128, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("[INFO] Model compiled!")

# -----------------------------
# 6️⃣ Train Model
# -----------------------------
print("[INFO] Starting training...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,            # reduce for faster training
    verbose=1
)

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
model.save("plant_disease_model.h5")
print("[INFO] Training done!")
print("[INFO] Model saved as plant_disease_model.h5")
