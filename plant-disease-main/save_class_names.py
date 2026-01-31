import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = r"C:\Users\nawyah11112004\Downloads\plant-disease-main (1)\plant-disease-main\dataset\train"

datagen = ImageDataGenerator(rescale=1.0/255)
generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

class_names = list(generator.class_indices.keys())

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("class_names.json created successfully!")
