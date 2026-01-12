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
}

void loop() {
  /**
   * Fetches iris prediction from Bridge and displays corresponding frame.
   * Refresh rate: 1 second.
   */
  String species;
  // bool ok = Bridge.call("predict_iris", 5.1, 3.5, 1.4, 0.2).result(species); // Setosa
  // bool ok = Bridge.call("predict_iris", 5.9, 2.8, 4.3, 1.3).result(species); // Versicolor
  bool ok = Bridge.call("predict_iris", 6.6, 3.0, 5.5, 2.0).result(species); // Virginica
  
  if (ok) {
    if (species == "setosa") loadFrame8x13(setosa);
    else if (species == "versicolor") loadFrame8x13(versicolor);
    else if (species == "virginica") loadFrame8x13(virginica);
    else loadFrame8x13(unknown);
  }
  delay(1000);
}
