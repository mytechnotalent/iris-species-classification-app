"""
FILE: main.py

DESCRIPTION:
  Loads a trained PyTorch model for iris classification and provides prediction functionality.

BRIEF:
  Provides functions to predict iris species based on sepal and petal measurements.
  Allows a microcontroller to call these functions via Bridge to display predictions.

AUTHOR: Kevin Thomas
CREATION DATE: January 11, 2026
UPDATE DATE: February 24, 2026
"""

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# Species and device configuration
# Map class indices to iris species names
SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Determine compute device: CUDA > MPS > CPU for optimal performance
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class Model(nn.Module):
    """
    Feedforward neural network for iris species classification.

    A two-layer feedforward network with optional dropout for regularization.
    Accepts 3 engineered features and outputs 3 class logits.

    ATTRIBUTES:
      fc1 (nn.Linear): First hidden layer (input -> h1).
      fc2 (nn.Linear): Second hidden layer (h1 -> h2).
      out (nn.Linear): Output layer (h2 -> num_classes).
      dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, in_features=3, h1=8, h2=8, out_features=3, dropout=0.0):
        """
        Initialize neural network layers.

        PARAMETERS:
          in_features (int): Number of input features (default: 3).
          h1 (int): Number of neurons in first hidden layer (default: 8).
          h2 (int): Number of neurons in second hidden layer (default: 8).
          out_features (int): Number of output classes (default: 3).
          dropout (float): Dropout rate, 0.0 = no dropout (default: 0.0).

        RETURNS:
          None
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass through the network with ReLU activation and dropout.

        PARAMETERS:
          x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        RETURNS:
          torch.Tensor: Output logits of shape (batch_size, out_features).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


# Load pre-trained model and data scaler
# Initialize model, load trained weights, and set to evaluation mode
model = Model().to(DEVICE)
model.load_state_dict(
    torch.load("/app/python/iris_model.pth", map_location=DEVICE, weights_only=True)
)
model.eval()
# Load fitted scaler used during model training for feature normalization
scaler = joblib.load("/app/python/iris_scaler.pkl")


# Private helper functions for predict_iris (in call order)
def _prepare_raw_features(
    sepal_dominance: float, petal_width: float, petal_length: float
) -> list:
    """
    Prepare raw features in lean_features order.

    PARAMETERS:
      sepal_dominance (float): Sepal dominance feature (0.0 or 1.0).
      petal_width (float): Petal width measurement in cm.
      petal_length (float): Petal length measurement in cm.

    RETURNS:
      list: Features in order [sepal_dominance, petal_width, petal_length].
    """
    return [sepal_dominance, petal_width, petal_length]


def _scale_and_tensorize(raw_features: list) -> torch.Tensor:
    """
    Scale features and convert to PyTorch tensor.

    PARAMETERS:
      raw_features (list): Raw feature values to scale.

    RETURNS:
      torch.Tensor: Scaled features as 2D tensor ready for model input.
    """
    scaled = scaler.transform([raw_features])[0]
    X = torch.tensor(scaled).float().to(DEVICE)
    return X.unsqueeze(0) if X.dim() == 1 else X


def _predict_class(X_tensor: torch.Tensor) -> int:
    """
    Get predicted class index from model logits.

    PARAMETERS:
      X_tensor (torch.Tensor): Scaled feature tensor.

    RETURNS:
      int: Predicted class index (0, 1, or 2).
    """
    logits = model(X_tensor)
    return logits.argmax(dim=1).item()


def predict_iris(
    sepal_dominance: float, petal_width: float, petal_length: float
) -> str:
    """
    Predict iris species from engineered features.

    Prepares features, scales them, and passes through the trained model to
    predict the iris species.

    PARAMETERS:
      sepal_dominance (float): 1.0 if sepal_length > 2*petal_length, else 0.0.
      petal_width (float): Petal width measurement in cm.
      petal_length (float): Petal length measurement in cm.

    RETURNS:
      str: Predicted iris species name or error message.
    """
    try:
        raw_features = _prepare_raw_features(sepal_dominance, petal_width, petal_length)
        X_tensor = _scale_and_tensorize(raw_features)
        predicted_class = _predict_class(X_tensor)
        return SPECIES_MAP[predicted_class]
    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Application interface setup
# Expose predict_iris function for direct microcontroller calls via Bridge protocol
Bridge.provide("predict_iris", predict_iris)

# Initialize web UI server on port 7000 for user interaction
ui = WebUI(port=7000)


# Private helper functions for on_predict (in call order)
def _extract_flower_measurements(data: dict) -> tuple:
    """
    Extract iris measurements from request data.

    PARAMETERS:
      data (dict): Request data containing flower measurements.

    RETURNS:
      tuple: (sepal_dominance, petal_width, petal_length) with 0.0 defaults.
    """
    return (
        data.get("sepal_dominance", 0.0),
        data.get("petal_width", 0.0),
        data.get("petal_length", 0.0),
    )


def _send_result_to_clients(species: str) -> None:
    """
    Send prediction result to web client and LED matrix display.

    PARAMETERS:
      species (str): Predicted iris species name.

    RETURNS:
      None
    """
    ui.send_message("prediction_result", {"species": species})
    Bridge.call("display_species", species)


def on_predict(client, data):
    """
    Handle iris species prediction request from web interface.

    Extracts measurements, runs prediction, and broadcasts result to web
    client and LED matrix display.

    PARAMETERS:
      client: The client connection requesting prediction.
      data (dict): Request data with sepal_dominance, petal_width, petal_length.

    RETURNS:
      None
    """
    sepal_dominance, petal_width, petal_length = _extract_flower_measurements(data)
    species = predict_iris(sepal_dominance, petal_width, petal_length)
    _send_result_to_clients(species)


# Script-level event handling and application initialization
# Register the on_predict handler to receive prediction requests via web socket
ui.on_message("predict", on_predict)

# Start the application main loop
App.run()
