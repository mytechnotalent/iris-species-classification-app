# Iris Species Classification App

**AUTHOR:** [Kevin Thomas](ket189@pitt.edu)
**CREATION DATE:** January 11, 2026  
**UPDATE DATE:** February 26, 2026  

Data Source: [HERE](https://github.com/mytechnotalent/I-MLP)

Classify iris flowers in real time with a PyTorch model and see predictions on an Arduino LED matrix by entering measurements via the web interface.

## Description

The App uses a Multi-Layer Perceptron (MLP) neural network trained on the classic Iris dataset to predict iris species from six input features: sepal length, sepal width, petal length, petal width, petal shape, and sepal dominance. The last two features are engineered: petal shape is the petal length / petal width category, and sepal dominance indicates whether the sepal length is more than twice the petal length. Users can enter measurements through a web interface, and the prediction is visualized on the 8 x 13 LED matrix with unique flower patterns for each species. Features are listed in order of importance as determined by their final Discriminative Score ($S$), which evaluates their ability to separate the classes by combining p-value strength ($P$), effect size ($E$), and normalized mutual information ($M$).

The `assets` folder contains the **frontend** components of the application, including the HTML, CSS, and JavaScript files that make up the web user interface. The `python` folder contains the application **backend** with model inference and WebUI handling. The Arduino sketch manages LED matrix display.

## Bricks Used

The Iris Species Classification App uses the following Bricks:

- `arduino:web_ui`: Brick to create a web interface for inputting iris measurements and displaying predictions.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)

### Software

- Arduino App Lab
- PyTorch (for neural network inference)

## How to Use the Example

### Clone the Example

1. Clone the example to your workspace.

### Run the App

1. Click the **Run** button in App Lab to start the application.
2. Open the App in your browser at `<UNO-Q-IP-ADDRESS>:7000`
3. Enter the six iris features:
   - **Sepal Length**: measurement in cm (e.g., `5.1`)
   - **Sepal Width**: measurement in cm (e.g., `3.5`)
   - **Petal Length**: measurement in cm (e.g., `1.4`)
   - **Petal Width**: measurement in cm (e.g., `0.2`)
   - **Petal Shape**: petal_length / petal_width category (e.g., `1.0`)
   - **Sepal Dominance**: `1.0` if sepal_length > 2×petal_length, else `0.0`
4. Click **Predict Species** to see the result

### Input Validation

The web interface validates that all inputs are proper floats:
- Integers are rejected (e.g., `5` must be entered as `5.0`)
- Text/strings are rejected
- Sepal Dominance must be `0.0` or `1.0`
- Valid format examples: `5.1`, `3.5`, `1.4`, `0.2`, `1.0`

### Example Measurements

Try these sample measurements to test each species prediction:

| Species    | Sepal Length | Sepal Width | Petal Length | Petal Width | Petal Shape | Sepal Dominance |
| ---------- | ------------ | ----------- | ------------ | ----------- | ----------- | --------------- |
| Setosa     | 5.1          | 3.5         | 1.4          | 0.2         | 1.0         | 1.0             |
| Versicolor | 5.9          | 2.8         | 4.3          | 1.3         | 0.0         | 0.0             |
| Virginica  | 6.3          | 3.3         | 5.5          | 2.0         | 0.0         | 0.0             |

**Note:** Sepal Dominance = `1.0` means the flower's sepal length is more than twice its petal length (characteristic of setosa).

## How it Works

Once the application is running, the device performs the following operations:

- **Serving the web interface and handling WebSocket communication.**

  The `web_ui` Brick provides the web server and WebSocket communication:

  ```python
  from arduino.app_bricks.web_ui import WebUI

  ui = WebUI()
  ui.on_message("predict", on_predict)
  ```

- **Loading the trained PyTorch model and scaler.**

  The application loads a pre-trained MLP model and StandardScaler for iris classification:

  ```python
  from arduino.app_utils import *
  import torch
  import torch.nn as nn
  import joblib

  model = Model().to(DEVICE)
  model.load_state_dict(torch.load("/app/python/iris_model.pth", map_location=DEVICE, weights_only=True))
  model.eval()

  scaler = joblib.load("/app/python/iris_scaler.pkl")
  ```

  The model and scaler are automatically loaded when the application starts and are ready to make predictions.

- **Making predictions based on input measurements.**

  The `predict_iris()` function takes six features, scales them, and returns the predicted species:

  ```python
  def predict_iris(sepal_length: float, sepal_width: float, petal_length: float,
                   petal_width: float, petal_shape: float, sepal_dominance: float) -> str:
      raw_features = [sepal_length, sepal_width, petal_length, petal_width,
                      petal_shape, sepal_dominance]
      scaled_features = scaler.transform([raw_features])[0]
      X_new = torch.tensor(scaled_features).float().to(DEVICE)
      logits = model(X_new)
      predicted_class = logits.argmax(dim=1).item()
      return SPECIES_MAP[predicted_class]
  ```

  The model outputs one of three species: `setosa`, `versicolor`, or `virginica`.

- **Handling web interface predictions and updating the LED matrix.**

  When a user submits measurements through the web interface:

  ```python
  def on_predict(client, data):
      species = predict_iris(data["sepal_length"], data["sepal_width"],
                             data["petal_length"], data["petal_width"],
                             data["petal_shape"], data["sepal_dominance"])
      ui.send_message("prediction_result", {"species": species})
      Bridge.call("display_species", species)
  ```

- **Displaying species patterns on the LED matrix.**

  The sketch receives the species and displays the corresponding pattern:

  ```cpp
  void display_species(String species) {
    if (species == "setosa") loadFrame8x13(setosa);
    else if (species == "versicolor") loadFrame8x13(versicolor);
    else if (species == "virginica") loadFrame8x13(virginica);
    else loadFrame8x13(unknown);
  }
  ```

The high-level data flow looks like this:

```
Web Browser Input → WebSocket → Python Backend → PyTorch Model → Bridge → LED Matrix
```

- **`ui = WebUI()`**: Initializes the web server that serves the HTML interface and handles WebSocket communication.

- **`ui.on_message("predict", on_predict)`**: Registers a WebSocket message handler that responds when the user submits measurements.

- **`ui.send_message("prediction_result", ...)`**: Sends prediction results to the web client in real-time.

- **`SPECIES_MAP`**: Dictionary mapping class indices to species names (0 = setosa, 1 = versicolor, 2 = virginica).

- **`DEVICE`**: Automatically selects the best available compute device (CUDA, MPS, or CPU).

- **`Model`**: A feedforward neural network class with two hidden layers (8 neurons each) and optional dropout for regularization.

- **`predict_iris()`**: Takes six float features (sepal length, sepal width, petal length, petal width, petal shape, sepal dominance), scales them using the fitted StandardScaler, runs inference through the model, and returns the predicted species name.

- **`Bridge.call("display_species", species)`**: Calls the Arduino function to update the LED matrix display.

### 🔧 Frontend (`index.html` + `app.js`)

The web interface provides a form for entering iris measurements with validation.

- **Socket.IO connection**: Establishes WebSocket communication with the Python backend through the `web_ui` Brick.

- **`socket.emit("predict", data)`**: Sends measurement data to the backend when the user clicks the predict button.

- **`socket.on("prediction_result", ...)`**: Receives prediction results and updates the UI accordingly.

- **`isValidFloat()`**: Validates that inputs are proper floats (rejects integers and strings).

### 🔧 Hardware (`sketch.ino`)

The Arduino code is focused on hardware management. It receives species names and displays them on the LED matrix.

- **`matrix.begin()`**: Initializes the matrix driver, making the LED display ready to show patterns.

- **`Bridge.begin()`**: Opens the serial communication bridge to the host Python® runtime.

- **`Bridge.provide("display_species", display_species)`**: Registers the display function to be callable from Python.

- **`display_species(String species)`**: Receives the predicted species and displays the corresponding 8 × 13 frame on the LED matrix.

- **`loop()`**: Once per second, calls the Python® function with iris measurements, selects the corresponding 8 × 13 frame (`setosa`, `versicolor`, `virginica`, or `unknown`), and shows it with `loadFrame8x13(frame)`.

- **`iris_frames.h`**: Header file that stores the pixel patterns for each iris species:
  - **Setosa**: Small diamond flower pattern
  - **Versicolor**: Medium cross-shaped flower pattern  
  - **Virginica**: Large starburst flower pattern
  - **Unknown**: Question mark for error cases

## Neural Network Architecture

The MLP model consists of:

- **fc1**: Input (6) → Output (8), ReLU activation
- **fc2**: Input (8) → Output (8), ReLU activation  
- **out**: Input (8) → Output (3), logits output

The model takes 6 input features (sepal length, sepal width, petal length, petal width, petal shape, sepal dominance) and outputs probabilities for 3 iris species. Features are scaled using StandardScaler before inference.

## Author

**Kevin Thomas**

- Creation Date: January 11, 2026
- Last Updated: February 26, 2026
