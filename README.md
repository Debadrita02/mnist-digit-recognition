# MNIST Handwritten Digit Recognition

## Project Overview
This project focuses on recognizing handwritten digits (0–9) using Machine Learning algorithms. It uses the MNIST dataset, which contains 70,000 grayscale images of handwritten digits.

A Streamlit web application allows users to select sample handwritten digit images and view the model's prediction.

## Dataset
- Dataset: MNIST
- Total images: 70,000
- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)

Each image contains 784 pixel values, which are scaled to the range 0–1 before model training.

## Machine Learning Models
Three classification algorithms were implemented and compared:

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Random Forest Classifier

The Random Forest model achieved the highest accuracy in the evaluated setup and was selected for the web application.

## Model Performance

| Model | Accuracy |
|---|---:|
| KNN | 94.63% |
| SVM | 91.70% |
| Random Forest | 95.03% |

*Note: Models were trained using the first 10,000 images from the training dataset. The test dataset was kept separate for evaluation.*

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
mnist-digit-recognition/
│
├── notebooks/
│   └── 01 MNSIT Digit Recognition.ipynb
│
├── sample_images/
├── app.py
├── best_mnist_model.pkl
└── README.md
```

## How to Run the Project

1. Install the required Python libraries:

```bash
pip install streamlit scikit-learn numpy pandas matplotlib joblib pillow
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. Open the local URL displayed in the terminal.

## Application Features
- Select sample handwritten digit images.
- Predict the digit using the trained Random Forest model.
- Compare the predicted digit with the actual label.
- View prediction probabilities.

## Limitations
- The model was trained on a subset of the available training images rather than all 60,000 images.
- Performance may vary for handwritten digits that differ significantly from the MNIST dataset.

## Author
Machine Learning Internship Project