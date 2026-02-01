// SPDX-FileCopyrightText: Copyright (C) Kevin Thomas
//
// SPDX-License-Identifier: MPL-2.0

const socket = io(`http://${window.location.host}`);

let errorContainer;

document.addEventListener('DOMContentLoaded', () => {
    errorContainer = document.getElementById('error-container');
    initSocketIO();
    initForm();
});

function initSocketIO() {
    socket.on('connect', () => {
        if (errorContainer) {
            errorContainer.style.display = 'none';
            errorContainer.textContent = '';
        }
    });

    socket.on('prediction_result', (message) => {
        showResult(message.species);
    });

    socket.on('disconnect', () => {
        if (errorContainer) {
            errorContainer.textContent = 'Connection to the board lost. Please check the connection.';
            errorContainer.style.display = 'block';
        }
    });
}

function initForm() {
    const form = document.getElementById('iris-form');
    form.addEventListener('submit', handleSubmit);
}

function isValidFloat(value) {
    const trimmed = value.trim();
    if (trimmed === '') return false;
    // Reject pure integers (must have decimal point)
    if (/^-?\d+$/.test(trimmed)) return false;
    // Accept floats with decimal point
    if (/^-?\d*\.\d+$/.test(trimmed)) return true;
    return false;
}

function handleSubmit(e) {
    e.preventDefault();

    const fields = [
        { id: 'sepal-dominance', errorId: 'sd-error' },
        { id: 'petal-width', errorId: 'pw-error' },
        { id: 'petal-length', errorId: 'pl-error' }
    ];

    let valid = true;

    fields.forEach(field => {
        const input = document.getElementById(field.id);
        const error = document.getElementById(field.errorId);

        if (!isValidFloat(input.value)) {
            input.classList.add('error');
            error.style.display = 'block';
            valid = false;
        } else {
            input.classList.remove('error');
            error.style.display = 'none';
        }
    });

    if (valid) {
        const data = {
            sepal_dominance: parseFloat(document.getElementById('sepal-dominance').value),
            petal_width: parseFloat(document.getElementById('petal-width').value),
            petal_length: parseFloat(document.getElementById('petal-length').value)
        };

        socket.emit('predict', data);
        document.getElementById('result').style.display = 'none';
    }
}

function showResult(species) {
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    resultText.textContent = `Predicted Species: ${species}`;
    resultDiv.style.display = 'block';
}
