function predictSpeed() {
    const selectedModel = document.getElementById('model-select').value;
    const checkpoint = document.getElementById('checkpoint').value;
    const timestamp = document.getElementById('timestamp').value;
    const traffic = document.getElementById('traffic').value;
    const intersection = document.getElementById('intersection').value;
    const roadHierarchy = document.getElementById('road-hierarchy').value;
    const latLongInput = document.getElementById('latLong').value;
    
    // Extract latitude and longitude from the input
    const [latitude, longitude] = latLongInput.split(',').map(coord => coord.trim());

    fetch('/predict_speed', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ selectedModel, checkpoint, timestamp, traffic, intersection, roadHierarchy, latitude, longitude })
    })
    .then(response => response.json())
    .then(data => {
        const predictionText = document.getElementById('prediction-text');

        if (data.success) {
            const predictedSpeed = parseInt(data.prediction);
            predictionText.innerHTML = `Predicted Speed: ${predictedSpeed} <br>`;

            // Check if the predicted speed is less than or equal to 60
            if (predictedSpeed <= 60) {
                predictionText.style.color = 'green';
                predictionText.innerHTML += ' <span style="font-weight: bold;">(Good Driver)</span>';
            } else {
                predictionText.style.color = 'red';
                predictionText.innerHTML += ' <span style="font-weigxht: bold;">(Bad Driver)</span>';
            }
        } else {
            predictionText.innerText = `Error: ${data.error}`;
            predictionText.style.color = 'red';
        }
    });

    // Prevent form submission
    return false;
}
