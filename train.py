import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import models, layers, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# Paths to your data
# -----------------------------

train_dir = r'D:\OneDrive\Documents\VGG-16\training_data'
validation_dir = r'D:\OneDrive\Documents\VGG-16\validation_data'

# -----------------------------
# Model Setup
# -----------------------------

image_size = 224  # Recommended for VGG16

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all layers except the last 4
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check trainable status
for layer in vgg_conv.layers:
    print(layer.name, layer.trainable)

# Build model
model = models.Sequential([
    vgg_conv,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # 5 classes including 'unknown'
])

model.summary()

# -----------------------------
# Data Generators
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Compile Model
# -----------------------------

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc']
)

# -----------------------------
# Checkpoint Callback
# -----------------------------

filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.keras"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max'
)
callbacks_list = [checkpoint]

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=35,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    verbose=1,
    callbacks=callbacks_list
)

# -----------------------------
# Plot Training Progress
# -----------------------------

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_loss.png')
plt.clf()

# -----------------------------
# Evaluation with Confusion Matrix
# -----------------------------

# Re-create validation generator with shuffle=False for evaluation
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Get ground truth
ground_truth = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Predict
predictions = model.predict(
    validation_generator,
    steps=validation_generator.samples // validation_generator.batch_size + 1,
    verbose=1
)
predicted_classes = np.argmax(predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(ground_truth, predicted_classes, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(ground_truth, predicted_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.clf()

# Misclassification summary
errors = np.where(predicted_classes != ground_truth)[0]
print(f"No of errors = {len(errors)}/{validation_generator.samples}")

print("✅ Training and evaluation complete.")
