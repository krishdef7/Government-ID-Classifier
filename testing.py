import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Paths
model_path = 'weights-improvement-02-0.98.keras'
test_dir = 'testing_data'  # 🔴 Change if your folder has a different name

try:
    print("🔵 Loading model...")
    model = load_model(model_path)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Prepare data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# Evaluation report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Accuracy summary
accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"\n✅ Overall Test Accuracy: {accuracy*100:.2f}%")

# Identify misclassified images
errors = np.where(predicted_classes != true_classes)[0]
print(f"\n❌ No of errors = {len(errors)}/{len(true_classes)}")
print("\nMisclassified images:")

for i in errors:
    print(f"Filename: {test_gen.filenames[i]} | True: {class_labels[true_classes[i]]} | Predicted: {class_labels[predicted_classes[i]]}")
