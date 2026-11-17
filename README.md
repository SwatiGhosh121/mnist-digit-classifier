# Handwritten Digit Recognition using Artificial Neural Network

An interactive web application that recognizes handwritten digits (0–9) using an Artificial Neural Network (ANN) trained on the MNIST dataset.

Users can draw a digit on a digital canvas and receive a prediction, confidence score, and probability distribution across all ten digits.

## Project Demo

**Live Application:** https://swatighosh121-mnist-digit-classifier-app-wmxfls.streamlit.app/git

**GitHub Repository:** https://github.com/SwatiGhosh121/mnist-digit-classifier

## Features

- Interactive drawing canvas.
- Handwritten digit recognition from 0 to 9.
- Real-time predictions using a trained ANN.
- Prediction confidence score.
- Probability distribution for all ten digits.
- Clear Canvas functionality.
- Image preprocessing and centre-of-mass alignment.
- Visualization of the processed 28 × 28 image.

## Model Architecture

The project uses a Sequential Artificial Neural Network built with TensorFlow/Keras.

| Layer | Configuration |
|---|---|
| Input | 28 × 28 grayscale image |
| Flatten | Converts image into a 1D vector |
| Hidden Layer | 128 neurons, ReLU activation |
| Output Layer | 10 neurons, Softmax activation |

### Model Training

- Dataset: MNIST Handwritten Digits
- Training samples: 60,000
- Testing samples: 10,000
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 10
- Reported test accuracy: 97.45%

## Image Preprocessing

Before passing a drawing to the model, the application performs the following operations:

1. Converts the canvas image into grayscale.
2. Detects and crops the handwritten digit.
3. Resizes the digit to fit within a 20 × 20 region.
4. Places it inside a 28 × 28 black image.
5. Applies centre-of-mass alignment.
6. Normalizes pixel values between 0 and 1.
7. Passes the processed image to the trained ANN.

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** TensorFlow, Keras
- **Numerical Computing:** NumPy, SciPy
- **Image Processing:** Pillow
- **Web Framework:** Streamlit
- **Dataset:** MNIST

## Project Structure

```text
mnist-digit-classifier/
│
├── model/
│   └── mnist_ann.keras
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SwatiGhosh121/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Future Improvements

- Improve recognition across different handwriting styles.
- Experiment with CNN architectures.
- Add prediction history.
- Improve image preprocessing and visualization.
- Compare ANN and CNN performance.

## Author

Developed as a machine learning and full-stack deployment project.

**Built with Python, TensorFlow and Streamlit.**