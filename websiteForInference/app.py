from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the selected model based on the dropdown choice
def load_model(selected_model):
    # Replace this with your actual model loading code
    if selected_model == 'LinearRegression':
        model_filename = 'LinearRegression.pkl'
    elif selected_model == 'DecisionTree':
        model_filename = 'DecisionTree.pkl'
    elif selected_model == 'RandomForest':
        model_filename = 'RandomForest.pkl'
    else:
        raise ValueError('Invalid model selection')

    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_speed', methods=['POST'])
def predict_speed():
    try:
        data = request.get_json()
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        selected_model = data['selectedModel']

        model = load_model(selected_model)

        speed_prediction = model.predict([[latitude, longitude]])

        response = {
            'success': True,
            'prediction': speed_prediction.tolist()[0]  # Convert NumPy array to Python list
        }
    except Exception as e:
        response = {
            'success': False,
            'error': str(e)
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
