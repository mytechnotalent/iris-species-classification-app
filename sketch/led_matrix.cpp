/*
 * FILE: led_matrix.cpp
 *
 * DESCRIPTION:
 *   LED matrix implementation for 8x13 air quality display.
 *   Provides functions to convert frame arrays and load them onto the matrix.
 *
 * BRIEF:
 *   Implements the LED matrix object and helper functions for frame handling.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: December 22, 2025
 * UPDATE DATE: December 22, 2025
 *
 * NOTES:
 *   Requires Arduino_LED_Matrix library.
 */

#include "led_matrix.h"

Arduino_LED_Matrix matrix;

void convertToFrameBuffer(const uint8_t src[8][13], uint32_t dest[4]) {
  for (int i = 0; i < 4; i++) dest[i] = 0;
  int bitPos = 0;
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 13; col++) {
      if (src[row][col]) {
        int idx = bitPos / 32;
        int shift = 31 - (bitPos % 32);
        dest[idx] |= (1UL << shift);
      }
      bitPos++;
    }
  }
}

void loadFrame8x13(const uint8_t frame[8][13]) {
  uint32_t frameBuffer[4];
  convertToFrameBuffer(frame, frameBuffer);
  matrix.loadFrame(frameBuffer);
}