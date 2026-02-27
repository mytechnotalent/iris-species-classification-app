/**
 * FILE: app.js
 *
 * DESCRIPTION:
 *   Web UI for iris species classification app.
 *   Handles form submission, validation, socket communication, and result display.
 *
 * BRIEF:
 *   Manages client-side interactions for iris prediction input and output.
 *   Uses WebSocket (Socket.IO) to communicate with Arduino backend.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: January 11, 2026
 * UPDATE DATE: February 26, 2026
 *
 * SPDX-FileCopyrightText: Copyright (C) Kevin Thomas
 * SPDX-License-Identifier: MPL-2.0
 */

// Initialize Socket.IO connection to server
const socket = io(`http://${window.location.host}`);

// Reference to error container element for connection status messages
let errorContainer;

/**
 * Initialize event listeners when DOM is fully loaded.
 */
document.addEventListener('DOMContentLoaded', () => {
    errorContainer = document.getElementById('error-container');
    _init_socket_events();
    _init_form_handler();
});

/**
 * Private helper to handle connection established event.
 */
function _handle_connect() {
    if (errorContainer) {
        errorContainer.style.display = 'none';
        errorContainer.textContent = '';
    }
}

/**
 * Private helper to handle prediction result received.
 *
 * PARAMETERS:
 *   message (object): Message object containing species prediction.
 */
function _handle_prediction_result(message) {
    _display_result(message.species);
}

/**
 * Private helper to handle connection lost event.
 */
function _handle_disconnect() {
    if (errorContainer) {
        errorContainer.textContent = 'Connection to the board lost. Please check the connection.';
        errorContainer.style.display = 'block';
    }
}

/**
 * Initialize Socket.IO event listeners for connection and messages.
 */
function _init_socket_events() {
    socket.on('connect', _handle_connect);
    socket.on('prediction_result', _handle_prediction_result);
    socket.on('disconnect', _handle_disconnect);
}

/**
 * Initialize form submission event handler.
 */
function _init_form_handler() {
    const form = document.getElementById('iris-form');
    form.addEventListener('submit', _on_form_submit);
}

/**
 * Validate if value is a valid floating-point number.
 *
 * Must have decimal point; rejects pure integers and empty strings.
 *
 * PARAMETERS:
 *   value (string): The value to validate.
 *
 * RETURN:
 *   boolean: true if valid float, false otherwise.
 */
function _is_valid_float(value) {
    const trimmed = value.trim();
    if (trimmed === '') return false;
    if (/^-?\d+$/.test(trimmed)) return false;
    return /^-?\d*\.\d+$/.test(trimmed);
}

/**
 * Private helper to validate all form input fields.
 *
 * Updates error display states for invalid fields.
 *
 * PARAMETERS:
 *   fields (array): Array of field objects with id and errorId.
 *
 * RETURN:
 *   boolean: true if all fields valid, false otherwise.
 */
function _validate_form_fields(fields) {
    let valid = true;
    fields.forEach(field => {
        const input = document.getElementById(field.id);
        const error = document.getElementById(field.errorId);
        if (!_is_valid_float(input.value)) {
            input.classList.add('error');
            error.style.display = 'block';
            valid = false;
        } else {
            input.classList.remove('error');
            error.style.display = 'none';
        }
    });
    return valid;
}

/**
 * Private helper to collect and parse form input values.
 *
 * RETURN:
 *   object: Object with sepal_length, sepal_width, petal_length, petal_width,
 *           petal_shape, sepal_dominance properties.
 */
function _collect_form_data() {
    return {
        sepal_length: parseFloat(document.getElementById('sepal-length').value),
        sepal_width: parseFloat(document.getElementById('sepal-width').value),
        petal_length: parseFloat(document.getElementById('petal-length').value),
        petal_width: parseFloat(document.getElementById('petal-width').value),
        petal_shape: parseFloat(document.getElementById('petal-shape').value),
        sepal_dominance: parseFloat(document.getElementById('sepal-dominance').value)
    };
}

/**
 * Handle form submission and validate before sending prediction request.
 *
 * PARAMETERS:
 *   e (event): The form submission event.
 */
function _on_form_submit(e) {
    e.preventDefault();
    const fields = [
        { id: 'sepal-length', errorId: 'sl-error' },
        { id: 'sepal-width', errorId: 'sw-error' },
        { id: 'petal-length', errorId: 'pl-error' },
        { id: 'petal-width', errorId: 'pw-error' },
        { id: 'petal-shape', errorId: 'ps-error' },
        { id: 'sepal-dominance', errorId: 'sd-error' }
    ];
    if (_validate_form_fields(fields)) {
        socket.emit('predict', _collect_form_data());
        document.getElementById('result').style.display = 'none';
    }
}

/**
 * Display prediction result to user on the web UI.
 *
 * PARAMETERS:
 *   species (string): The predicted iris species name.
 */
function _display_result(species) {
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    resultText.textContent = `Predicted Species: ${species}`;
    resultDiv.style.display = 'block';
}
