import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# ── CONFIGURATION ───────────────────────────────────
DATASET_DIR = os.path.join('archive', 'PlantVillage')
IMG_SIZE = (128, 128)  # Optimized size for fast CPU training
BATCH_SIZE = 32
EPOCHS = 8
MODEL_PATH = 'plant_model.h5'
CLASS_JSON = 'class_indices.json'

print("TensorFlow Version:", tf.__version__)
print("Checking dataset directory:", DATASET_DIR)

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' not found. Please ensure the dataset is extracted under archive/PlantVillage.")

# ── LOAD DATASET ────────────────────────────────────
# image_dataset_from_directory is highly optimized and returns tf.data.Dataset
print("Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Get class names
class_names = train_ds.class_names
print("Detected classes:", class_names)

# Save class indices to json
class_indices = {i: name for i, name in enumerate(class_names)}
with open(CLASS_JSON, 'w') as f:
    json.dump(class_indices, f, indent=4)
print(f"Saved class indices mapping to '{CLASS_JSON}'")

# Configure dataset for performance (prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ── BUILD LIGHTWEIGHT CNN ───────────────────────────
# Optimized custom CNN architecture under 100k parameters for fast CPU training
model = models.Sequential([
    # Rescaling layer to normalize pixels to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Conv Block 1
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Conv Block 2
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Conv Block 3
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Flattening & Classification
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()

# ── COMPILE AND TRAIN ───────────────────────────────
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Starting training for {EPOCHS} epochs on CPU/GPU...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ── SAVE MODEL ──────────────────────────────────────
print(f"Saving trained model to '{MODEL_PATH}'...")
model.save(MODEL_PATH)
print("Model saved successfully! Ready for prediction.")
