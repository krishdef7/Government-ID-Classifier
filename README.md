# Government-ID-Classifier  

A deep learning project for **automatic classification of government ID cards** using transfer learning with **VGG16**. The model is fine-tuned on a custom dataset sourced from **Roboflow** and achieves an impressive **97% accuracy**.  

---

## 🚀 Features
- Uses **VGG16** pretrained on ImageNet.  
- **Transfer Learning**: all layers frozen except the last 4 for fine-tuning.  
- **Data Augmentation**: applied for better generalization.  
- **High Accuracy**: 97% on test data.  
- Training, validation, and testing pipelines included (`train.py`, `testing.py`).  
- Generates accuracy/loss plots and a confusion matrix for analysis.  

---

## 📂 Repository Structure
```
Government-ID-Classifier/
├── train.py              # Training script (VGG16 + fine-tuning)
├── testing.py            # Testing script for inference and evaluation
├── requirements.txt      # Project dependencies
├── models/               # Directory to save trained weights
├── training_data/        # Training dataset 
├── validation_data/      # Validation dataset
└── testing_data/         # Testing dataset
```

## 🏋️ Training

Run the training script:  
```bash
python train.py
```

- Uses **VGG16 backbone** with last 4 layers trainable.  
- Trains for 35 epochs with data augmentation (rotation, shift, flip, etc.).  
- Saves best weights as `.keras` files inside `models/`.  
- Outputs plots:  
  - `model_accuracy.png`  
  - `model_loss.png`  
  - `confusion_matrix.png`  

---

## 🧪 Testing / Inference

Run the testing script:  
```bash
python testing.py
```

- Loads the best saved weights (e.g., `weights-improvement-02-0.98.keras`).  
- Evaluates accuracy on unseen test images.  
- Prints **classification report** and lists misclassified images.  

---

## 📊 Results

- **Accuracy**: ~97%  
- **Precision/Recall/F1**: High across all classes  
- **Confusion Matrix**: Saved as `confusion_matrix.png`  
- Example evaluation snippet:
  ```
  ✅ Overall Test Accuracy: 97.00%
  ```

---

## 📜 License
This project is open-source. You may use, modify, and distribute it for research and educational purposes.  

---

📌 *Future Improvements*:  
- Add support for additional ID document types.  
- Deploy as a web app using Flask/FastAPI.  
- Experiment with newer architectures (e.g., EfficientNet, ViT).  
