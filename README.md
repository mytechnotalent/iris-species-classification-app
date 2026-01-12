# Iris Species Classification App

The Iris Species Classification App uses a trained PyTorch model to classify iris flowers and display the predicted species on the Arduino UNO Q LED matrix using distinct flower patterns for Setosa, Versicolor, or Virginica.

## Description

The App uses a Multi-Layer Perceptron (MLP) neural network trained on the classic Iris dataset to predict iris species from four input measurements: sepal length, sepal width, petal length, and petal width. The prediction is visualized on the 8 x 13 LED matrix with unique flower patterns for each species.

The Python® script handles model loading and inference using PyTorch, while the Arduino sketch manages LED matrix display and polling. The Router Bridge enables communication between the Python environment and the microcontroller.

## Bricks Used

**This example does not use any Bricks.** It shows direct Router Bridge communication between Python® and Arduino.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)

### Software

- Arduino App Lab
- PyTorch (for neural network inference)

**Note:** You can also run this example using your Arduino UNO Q as a Single Board Computer (SBC) using a [USB-C® hub](https://store.arduino.cc/products/usb-c-to-hdmi-multiport-adapter-with-ethernet-and-usb-hub) with a mouse, keyboard and display attached.

## How to Use the Example

### Clone the Example

1. Clone the example to your workspace.

### Configure Iris Measurements (Optional)

If you want to test with different iris measurements, update the values in `sketch.ino`:

```cpp
Bridge.call("predict_iris", 5.1, 3.5, 1.4, 0.2).result(species);
```

The four parameters represent:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

### Run the App

1. Click the **Run** button in App Lab to start the application.

## How it Works

Once the application is running, the device performs the following operations:

- **Loading the trained PyTorch model.**

  The application loads a pre-trained MLP model for iris classification:

  ```python
  from arduino.app_utils import *
  import torch
  import torch.nn as nn

  model = Model().to(DEVICE)
  model.load_state_dict(torch.load("/app/python/iris_model.pth", map_location=DEVICE))
  model.eval()
  ```

  The model is automatically loaded when the application starts and is ready to make predictions.

- **Making predictions based on input measurements.**

  The `predict_iris()` function takes four measurements and returns the predicted species:

  ```python
  def predict_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
      features = [sepal_length, sepal_width, petal_length, petal_width]
      X_new = torch.tensor(features).float().to(DEVICE)
      logits = model(X_new)
      predicted_class = logits.argmax(dim=1).item()
      return SPECIES_MAP[predicted_class]
  ```

  The model outputs one of three species: `setosa`, `versicolor`, or `virginica`.

- **Exposing prediction function to the microcontroller.**

  The Router Bridge makes the prediction function callable from the Arduino:

  ```python
  Bridge.provide("predict_iris", predict_iris)
  ```

- **Polling for predictions from the Arduino.**

  The Arduino sketch calls the Python function once every second:

  ```cpp
  Bridge.call("predict_iris", 5.1, 3.5, 1.4, 0.2).result(species);
  ```

  This allows real-time classification based on the provided measurements.

- **Displaying species patterns on the LED matrix.**

  The sketch maps each species to corresponding visual patterns:

  ```cpp
  if (species == "setosa") loadFrame8x13(setosa);
  else if (species == "versicolor") loadFrame8x13(versicolor);
  else if (species == "virginica") loadFrame8x13(virginica);
  else loadFrame8x13(unknown);
  ```

The high-level data flow looks like this:

```
Input Measurements → PyTorch Model → Router Bridge → MCU loop() → LED Matrix
```

## Understanding the Code

Here is a brief explanation of the application components:

### 🔧 Backend (`main.py`)

The Python® code serves as the system's inference engine, handling model loading and predictions using PyTorch.

- **`SPECIES_MAP`**: Dictionary mapping class indices to species names (0 = setosa, 1 = versicolor, 2 = virginica).

- **`DEVICE`**: Automatically selects the best available compute device (CUDA, MPS, or CPU).

- **`Model`**: A feedforward neural network class with two hidden layers (8 neurons each) and optional dropout for regularization.

- **`predict_iris()`**: Takes four float measurements (sepal length, sepal width, petal length, petal width), runs inference through the model, and returns the predicted species name.

- **`model.load_state_dict()`**: Loads the pre-trained model weights from `iris_model.pth`.

- **`Bridge.provide(...)`**: Makes `predict_iris` callable from the microcontroller, creating the communication link.

- **`App.run()`**: Starts the Router Bridge runtime that enables the Python®-Arduino communication.

### 🔧 Hardware (`sketch.ino`)

The Arduino code is focused on hardware management. It requests predictions and displays them.

- **`matrix.begin()`**: Initializes the matrix driver, making the LED display ready to show patterns.

- **`Bridge.begin()`**: Opens the serial communication bridge to the host Python® runtime.

- **`loop()`**: Once per second, calls the Python® function with iris measurements, selects the corresponding 8 × 13 frame (`setosa`, `versicolor`, `virginica`, or `unknown`), and shows it with `loadFrame8x13(frame)`.

- **`iris_frames.h`**: Header file that stores the pixel patterns for each iris species:
  - **Setosa**: Small diamond flower pattern
  - **Versicolor**: Medium cross-shaped flower pattern  
  - **Virginica**: Large starburst flower pattern
  - **Unknown**: Question mark for error cases

### 🔧 Model Training (`I-MLP.ipynb`)

The Jupyter notebook contains the complete model training pipeline:

- **Data Loading**: Loads the Iris dataset from `iris.csv`
- **Data Preprocessing**: Splits data into training and testing sets
- **Model Definition**: Defines the MLP architecture
- **Training Loop**: Trains the model using cross-entropy loss and Adam optimizer
- **Model Export**: Saves the trained weights to `iris_model.pth`

## Neural Network Architecture

The MLP model consists of:

- **fc1**: Input (4) → Output (8), ReLU activation
- **fc2**: Input (8) → Output (8), ReLU activation  
- **out**: Input (8) → Output (3), Softmax activation

The model takes 4 input features (sepal/petal measurements) and outputs probabilities for 3 iris species.

## Author

**Kevin Thomas**

- Creation Date: January 11, 2026
- Last Updated: January 11, 2026
