function predictSpeed() {
    const latLongInput = document.getElementById('latLong').value;
    const selectedModel = document.getElementById('modelSelect').value;
    
    // Extract latitude and longitude from the input
    const [latitude, longitude] = latLongInput.split(',').map(coord => coord.trim());

    fetch('/predict_speed', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ latitude, longitude, selectedModel })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('prediction').innerText = `Predicted Speed: ${data.prediction.toFixed(2)}`;
        } else {
            document.getElementById('prediction').innerText = `Error: ${data.error}`;
        }
    });

    // Prevent form submission
    return false;
}
