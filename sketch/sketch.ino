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
 * UPDATE DATE: February 24, 2026
 */

#include <Arduino_RouterBridge.h>
#include "iris_frames.h"
#include "led_matrix.h"

/**
 * Private helper function to load iris frame by species name.
 *
 * PARAMETERS:
 *   species (String): The predicted species name.
 *
 * RETURN:
 *   void
 */
void _load_species_frame(String species)
{
  if (species == "setosa")
    loadFrame8x13(setosa);
  else if (species == "versicolor")
    loadFrame8x13(versicolor);
  else if (species == "virginica")
    loadFrame8x13(virginica);
  else
    loadFrame8x13(unknown);
}

/**
 * Initialize LED matrix and Bridge communication.
 *
 * RETURN:
 *   void
 */
void setup()
{
  matrix.begin();
  matrix.clear();
  Bridge.begin();
  Bridge.provide("display_species", display_species);
}

/**
 * Main loop waiting for prediction updates from Python backend.
 *
 * RETURN:
 *   void
 */
void loop()
{
  delay(100);
}

/**
 * Display predicted iris species on LED matrix.
 *
 * PARAMETERS:
 *   species (String): The predicted iris species name.
 *
 * RETURN:
 *   void
 */
void display_species(String species)
{
  _load_species_frame(species);
}
