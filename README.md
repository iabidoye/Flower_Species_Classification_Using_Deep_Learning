# Flower_Species_Classification_Using_Deep_Learning
A deep learning-based approach for classifying flower species using CNN models. This repository includes data preprocessing, hyperparameter tuning, and training multiple CNN architectures to optimize classification accuracy. The repository contains a dataset, Jupyter notebook, and a detailed report on model performance.

# Flower Species Classification Using Deep Learning

## 📌 Overview
This project implements **Convolutional Neural Networks (CNNs)** to classify flower species based on image datasets. Various deep learning models and hyperparameter tuning techniques were used to improve accuracy, using TensorFlow and Keras.

## 📂 Project Structure
- **`Flower_Species_Image_Classification_Submission.ipynb`** - Jupyter Notebook containing data preprocessing, CNN model training, and evaluation.
- **`Flower_classification.pdf`** - A detailed scientific report on the flower classification models used.
- **`figures/`** - Directory containing training accuracy/loss plots and confusion matrices.

## 🛠️ Technologies Used
- **Python (TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn)**
- **Jupyter Notebook**
- **CNN Architectures:**
  - Convolutional layers with different kernel sizes
  - Pooling layers (MaxPooling2D, AveragePooling2D)
  - Dropout layers for overfitting control
  - Batch Normalization
  - Fully connected layers with Softmax activation

## 🔹 Key Features
### **1. Data Preprocessing & Augmentation**
- Image normalization and resizing to **(256,256,3)**.
- Data augmentation using **rotation, flipping, zooming, and shearing**.
- One-hot encoding for multi-class classification.

### **2. CNN Model Training & Optimization**
- Multiple CNN architectures tested for accuracy improvement.
- Tuned hyperparameters: **batch size, kernel size, pooling methods, dropout rate, and activation functions**.
- **Adam optimizer (learning rate 0.001) used for efficient convergence**.

### **3. Model Evaluation & Performance Metrics**
- Accuracy and loss tracking across **8 different CNN models**.
- **Confusion matrices** to evaluate misclassification.
- **Classification reports** for precision, recall, and F1-score.
- Comparison of different architectures and **best model selection**.

## 📊 Results & Insights
- **Best Model:** **CNN Model 1A (Batch size = 128)** achieved **78% accuracy**.
- The **original CNN structure** outperformed other variations in classifying flower species.
- **Batch normalization** improved accuracy while **average pooling reduced performance**.

## 🚀 Getting Started
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/iabidoye/Flower_Species_Classification_Using_Deep_Learning.git
   cd Flower_Prediction
2. Install required dependencies:
   ```bash
   pip install tensorflow keras numpy opencv-python matplotlib seaborn
3. Open the Jupyter Notebook and run the analysis:
   ```bash
   jupyter notebook Flower_Species_Image_Classification_Submission.ipynb

🤝 Contribution
Contributions are welcome! If you have improvements or new models, feel free to submit a pull request.

📧 Contact
For inquiries or collaborations, please reach out.

