# AgeEmotionDetection

This repository contains files and scripts for a machine learning project that detects age, emotion, and facial recognition. The repository includes pre-trained models and Python scripts for training new models. Below is an overview of each file and step-by-step instructions on using the scripts.

## Requirements

### **Programming Language and Version**
- **Python 3.8+** Ensure Python 3.8 or newer is installed to support the libraries and frameworks used in this project.
- **IDE** Some files are written as Jupyter Notebook (.ipynb) and Python files (.py). It is advised to have an IDE (VS Code or Anaconda Navigator) to handle both file type python execution. 

### **Libraries**
The following Python libraries are required for this project:
- **TensorFlow, Keras, OpenCV, Pandas, NumPy, Scikit-learn, OS, face_recognition**

## Datasets Used

1. **https://www.kaggle.com/datasets/msambare/fer2013**
   - Used for training the CNN model for emotion detection.  
   - Contains 35,887 grayscale images of 48x48 pixels, classified into seven emotions:  
     - **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.  
   - Widely used for facial expression recognition research.  

2. **https://chalearnlap.cvc.uab.cat/dataset/26/description/**  
   - Used for training the CNN model for age detection.  
   - Contains over 7,000 images of faces with real and apparent age labels.  
   - Images are labeled with the perceived age by human annotators as well as the actual age of the subject.  
   - Ideal for developing and testing age prediction models.

## Files Overview

### **Machine Learning Project.ipynb**
- An initial notebook by Minh Nguyen for facial recognition using reference images.
- Demonstrates foundational concepts for image-based facial recognition.

### **TrainEmotionDetector.py**
- Python script used to train the CNN model for emotion detection.
- Produces two key outputs:
  - `machine_learning_model.weights.h5`: Model weights for emotion detection.
  - `machine_learning_model.json`: Model architecture in JSON format.

### **machine_learning_model.weights.h5**
- Pre-trained model weights for emotion detection.

### **machine_learning_model.json**
- JSON file containing the architecture of the emotion detection CNN model.

### **Age-Prediction.ipynb**
- Notebook for training a CNN model for age detection.
- Produces the `age_detected_model.keras` file upon training.

### **age_detected_model.keras**
- Pre-trained model file for age detection.

### **Final_test_emotion_facial.ipynb**
- A notebook dedicated to testing emotion detection and facial recognition using:
  - Emotion detection model and weights.
  - Reference images for recognition.

### **FaceReg_Age_Emotion_Detection_final.ipynb**
- Combines facial recognition, age detection, and emotion detection in one script.
- Utilizes:
  - Emotion detection model and weights.
  - Age detection model for predictions.

### **haarcascades Folder**
- Contains Haar Cascade XML files for face detection, essential for preprocessing input images.

### **Reference Images**
- `reference1.jpg`: Minh Nguyen.
- `reference2.jpg`: Vishnu Muthyalu.
- `reference3.jpg`: Varun Prasad.
- Used for facial recognition in conjunction with the above scripts.

## How to Use

### **Step 1: Testing Age, Emotion, and Facial Recognition**
To test the pre-trained models with the combined functionality:
1. Open the `FaceReg_Age_Emotion_Detection_final.ipynb` notebook in your preferred Python environment (e.g., Jupyter Notebook, Google Colab).
2. Ensure the following files are in the same directory:
   - `machine_learning_model.weights.h5`
   - `machine_learning_model.json`
   - `age_detected_model.keras`
   - Reference images and Haar Cascade files.
3. Execute the cells in sequence. The notebook:
   - Recognizes faces using reference images.
   - Predicts age and emotion for detected faces.

### **Step 2: Training a New Emotion Detection Model**
To train a new model for emotion detection:
1. Open the `TrainEmotionDetector.py` file.
2. Replace the dataset path with your training dataset (e.g., FER-2013).
3. Run the script to train the model.
4. After training, the following files will be generated:
   - `machine_learning_model.weights.h5`: Updated weights.
   - `machine_learning_model.json`: Updated model architecture.

### **Step 3: Training a New Age Detection Model**
To train a new model for age detection:
1. Open the `Age-Prediction.ipynb` notebook.
2. Replace the dataset path with your training dataset for age detection.
3. Train the CNN model by executing the cells.
4. After training, the updated model will be saved as `age_detected_model.keras`.


