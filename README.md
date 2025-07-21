# Emotion Detector using SVM

This project is a facial **Emotion Detection System** that uses a **Support Vector Machine (SVM)** classifier to detect emotions from grayscale facial images. The application includes a **Tkinter-based GUI** that allows users to upload an image and get emotion predictions in real-time.

## 📌 Features

- Detects 7 emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`
- Uses SVM with hyperparameter tuning (`GridSearchCV`)
- Image preprocessing and resizing to (32x32)
- GUI built with Tkinter for easy interaction
- Model training with dataset directory structure

## 🧠 Model

- **Algorithm**: SVM (`RBF` kernel)
- **Hyperparameters** tuned using GridSearchCV
- **Pipeline** includes StandardScaler and SVM
- Model accuracy printed after training and saved using `joblib`

## 🖼️ Dataset Structure

Place your dataset in the following folder structure:

```
Emotion_detector/
│
└───dataset/
    ├───angry/
    ├───disgust/
    ├───fear/
    ├───happy/
    ├───neutral/
    ├───sad/
    └───surprise/
```

Each folder should contain grayscale face images of the respective emotion.

## 🔧 Installation

1. Clone the repository or download the code.
2. Install the required Python libraries:

```bash
pip install numpy opencv-python scikit-learn Pillow joblib
```

3. Place your dataset in `E:\Emotion_detector\dataset\` or modify the path in the script.

## 🚀 How to Run

```bash
python your_script_name.py
```

This will:
- Load a pre-trained model if available
- Else, train a new SVM model on your dataset
- Launch a GUI to upload and predict emotion from an image

## 🛠️ Main Files

- `load_data()`: Loads images from folders, resizes, and flattens them
- `train_model()`: Trains and saves the best model
- `EmotionDetector`: GUI class to handle image upload and prediction
- `svm_emotion_model.pkl`: Saved trained model

## 🧪 Example Prediction Flow

1. User uploads an image.
2. Image is converted to grayscale and resized to (32, 32).
3. Flattened image is passed to the SVM model.
4. Predicted emotion is shown in the GUI.

## 📎 Dependencies

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- Pillow
- Joblib
- Tkinter (usually pre-installed with Python)

## 📌 Notes

- Recommended image input size is **32x32 grayscale**.
- If your dataset is large, consider adjusting the `limit_per_class` to avoid long training times.
- You can retrain the model by deleting the `svm_emotion_model.pkl` file.


