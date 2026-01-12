"""
FILE: inference.py

DESCRIPTION:
  Loads a trained PyTorch model for iris classification and provides prediction functionality.

BRIEF:
  Provides functions to predict iris species based on sepal and petal measurements.
  Allows a microcontroller to call these functions via Bridge to display predictions.

AUTHOR: Kevin Thomas
CREATION DATE: January 11, 2026
UPDATE DATE: January 11, 2026
"""

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import torch
import torch.nn as nn
import torch.nn.functional as F

# Species mapping
SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Set device
if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
else:
  DEVICE = torch.device("cpu")


class Model(nn.Module):
  """
  A feedforward neural network with two hidden layers and optional dropout.
  """

  def __init__(self, in_features=4, h1=8, h2=8, out_features=3, dropout=0.0):
    """
    Initializes the neural network layers.

    PARAMETERS:
      in_features (int): Number of input features.
      h1 (int): Number of neurons in the first hidden layer.
      h2 (int): Number of neurons in the second hidden layer.
      out_features (int): Number of output features.
      dropout (float): Dropout rate (0.0 = no dropout).

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
    Defines the forward pass of the neural network.

    PARAMETERS:
      x (torch.Tensor): Input tensor.

    RETURNS:
      torch.Tensor: Output tensor after passing through the network.
    """
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.out(x)
    return x


# Load the model
model = Model().to(DEVICE)
model.load_state_dict(torch.load("/app/python/iris_model.pth", map_location=DEVICE))
model.eval()


def predict_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
  """
  Predicts the iris species based on provided measurements.

  PARAMETERS:
    sepal_length (float): Sepal length measurement.
    sepal_width (float): Sepal width measurement.
    petal_length (float): Petal length measurement.
    petal_width (float): Petal width measurement.

  RETURNS:
    str: Predicted iris species (e.g., "setosa", "versicolor", "virginica") or error message.
  """
  try:
    features = [sepal_length, sepal_width, petal_length, petal_width]
    X_new = torch.tensor(features).float().to(DEVICE)
    if X_new.dim() == 1:
      X_new = X_new.unsqueeze(0)
    logits = model(X_new)
    predicted_class = logits.argmax(dim=1).item()
    return SPECIES_MAP[predicted_class]
  except Exception as e:
    return f"Prediction Error: {str(e)}"


# Expose function to microcontroller via Bridge
Bridge.provide("predict_iris", predict_iris)

# Initialize WebUI on port 7000
ui = WebUI(port=7000)


def on_predict(client, data):
  """
  Handles prediction request from web interface.

  PARAMETERS:
    client: The client connection.
    data (dict): Contains sepal_length, sepal_width, petal_length, petal_width.

  RETURNS:
    None
  """
  sepal_length = data.get("sepal_length", 0.0)
  sepal_width = data.get("sepal_width", 0.0)
  petal_length = data.get("petal_length", 0.0)
  petal_width = data.get("petal_width", 0.0)

  species = predict_iris(sepal_length, sepal_width, petal_length, petal_width)

  # Send result back to web client
  ui.send_message("prediction_result", {"species": species})

  # Also update the LED matrix via Bridge
  Bridge.call("display_species", species)


# Handle socket messages
ui.on_message("predict", on_predict)

# Run app
App.run()