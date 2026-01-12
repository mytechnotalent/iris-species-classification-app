/**
 * FILE: sketch.ino
 *
 * DESCRIPTION:
 *   Displays iris species prediction on an 8x13 LED matrix using Arduino.
 *
 * BRIEF:
 *   Loads frames representing iris species on the matrix.
 *   Uses Bridge to fetch prediction from Python backend.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: January 11, 2026
 * UPDATE DATE: January 11, 2026
 */

#include <Arduino_RouterBridge.h>
#include "iris_frames.h"
#include "led_matrix.h"

void setup() {
  /**
   * Initializes LED matrix and Bridge.
   */
  matrix.begin();
  matrix.clear();
  Bridge.begin();
  Bridge.provide("display_species", display_species);
}

void loop() {
  /**
   * Waits for display_species calls from Python backend.
   * LED matrix is updated when prediction is received from web UI.
   */
  delay(100);
}

void display_species(String species) {
  /**
   * Displays the predicted iris species on the LED matrix.
   *
   * PARAMETERS:
   *   species (String): The predicted species name.
   */
  if (species == "setosa") loadFrame8x13(setosa);
  else if (species == "versicolor") loadFrame8x13(versicolor);
  else if (species == "virginica") loadFrame8x13(virginica);
  else loadFrame8x13(unknown);
}
