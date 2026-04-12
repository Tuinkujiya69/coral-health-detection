"""
Coral Health Detection - Improved Training Script
Uses MobileNetV2 Transfer Learning + Data Augmentation + Fine-Tuning
Expected accuracy: 85-92% (vs original 75%)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# =======================
# CONFIGURATION
# =======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 10
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "model/coral_model.h5"
CLASSES = ["Bleached Coral", "Dead Coral", "Healthy Coral"]  # ← 3 classes

# =======================
# STEP 1: DATA AUGMENTATION
# =======================
print("\n[STEP 1] Setting up data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    shear_range=0.1,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Classes found: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Save class labels for app.py
os.makedirs("model", exist_ok=True)
with open("model/class_labels.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("Class labels saved to model/class_labels.json")

# =======================
# STEP 2: BUILD MODEL
# =======================
print("\n[STEP 2] Building MobileNetV2 transfer learning model...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =======================
# STEP 3: PHASE 1 TRAINING
# =======================
print(f"\n[STEP 3] Phase 1 Training — {EPOCHS_PHASE1} epochs (base model frozen)...")

callbacks_phase1 = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    callbacks.ModelCheckpoint("model/best_phase1.h5", save_best_only=True, monitor="val_accuracy")
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks_phase1
)

print(f"\nPhase 1 best val accuracy: {max(history1.history['val_accuracy']):.4f}")

# =======================
# STEP 4: PHASE 2 FINE-TUNING
# =======================
print(f"\n[STEP 4] Phase 2 Fine-Tuning — unfreezing top 40 layers...")

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    callbacks.ModelCheckpoint("model/best_phase2.h5", save_best_only=True, monitor="val_accuracy")
]

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2
)

print(f"\nPhase 2 best val accuracy: {max(history2.history['val_accuracy']):.4f}")

# =======================
# STEP 5: SAVE FINAL MODEL
# =======================
print(f"\n[STEP 5] Saving final model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")

# =======================
# STEP 6: PLOT TRAINING CURVES
# =======================
print("\n[STEP 6] Plotting training curves...")

all_acc     = history1.history["accuracy"]     + history2.history["accuracy"]
all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
all_loss    = history1.history["loss"]         + history2.history["loss"]
all_val_loss= history1.history["val_loss"]     + history2.history["val_loss"]
phase_boundary = len(history1.history["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(all_acc,     label="Train Accuracy", color="#00b4d8", linewidth=2)
ax1.plot(all_val_acc, label="Val Accuracy",   color="#f77f00", linewidth=2)
ax1.axvline(x=phase_boundary, color="gray", linestyle="--", label="Fine-tuning starts")
ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(all_loss,     label="Train Loss", color="#00b4d8", linewidth=2)
ax2.plot(all_val_loss, label="Val Loss",   color="#f77f00", linewidth=2)
ax2.axvline(x=phase_boundary, color="gray", linestyle="--", label="Fine-tuning starts")
ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model/training_curves.png", dpi=150)
print("Training curves saved to model/training_curves.png")
plt.show()

# =======================
# FINAL EVALUATION
# =======================
print("\n[FINAL] Evaluating on validation set...")
loss, acc = model.evaluate(val_generator, verbose=1)
print(f"\n✅ Final Validation Accuracy: {acc * 100:.2f}%")
print(f"✅ Final Validation Loss: {loss:.4f}")
