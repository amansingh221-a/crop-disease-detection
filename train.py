import tensorflow as tf
import json
import os

# 1. Basic settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5

# 2. Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 3. Load validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 4. Get class names
class_names = train_data.class_names

print("Disease Classes:", class_names)
print("Total Classes:", len(class_names))

# 5. Create models folder
os.makedirs("models", exist_ok=True)

# 6. Save class names
with open("models/classes.json", "w") as f:
    json.dump(class_names, f)

# 7. Improve data loading
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

val_data = val_data.prefetch(
    buffer_size=AUTOTUNE
)

# 8. Create CNN model
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, 3, activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# 9. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 10. Display model structure
model.summary()

# 11. Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# 12. Save trained model
model.save("models/crop_model.keras")

print("Training Completed Successfully!")
print("Model Saved!")