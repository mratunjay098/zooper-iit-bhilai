from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the selected model based on the dropdown choice
def load_model(selected_model):
    # Replace this with your actual model loading code
    if selected_model == 'LinearRegressionLatLong':
        model_filename = 'models/linearRegressionLatLong.pkl'
    elif selected_model == 'DecisionTreeLatLong':
        model_filename = 'models/decisionTreeLatLong.pkl'
    elif selected_model == 'RandomForestLatLong':
        model_filename = 'models/randomForestLatLong.pkl'
    elif selected_model == 'LinearRegressionTSLatLong':
        model_filename = 'models/linearRegressionTSLatLong.pkl'
    elif selected_model == 'DecisionTreeTSLatLong':
        model_filename = 'models/decisionTreeTSLatLong.pkl'
    elif selected_model == 'LinearRegressionTSIRHLatLong':
        model_filename = 'models/linearRegressionTSIRHLatLong.pkl'
    elif selected_model == 'DecisionTreeTSIRHLatLong':
        model_filename = 'models/decisionTreeTSIRHLatLong.pkl'
    elif selected_model == 'RandomForestTSIRHLatLong':
        model_filename = 'models/randomForestTSIRHLatLong.pkl'
    else:
        raise ValueError('Invalid model selection')
    print(f"Attempting to load model from file: {os.path.abspath(model_filename)}")
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Successfully loaded model: {model}")
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
        if data['checkpoint']:
            checkpoint = data['checkpoint']
        if data['timestamp']:
            timestamp = data['timestamp']
        if data['traffic']:
            traffic = data['traffic']
        if data['intersection']:
            intersection = data['intersection']
        if data['roadHierarchy']:
            road_hierarchy = data['roadHierarchy']

        model = load_model(selected_model)

        if selected_model in ['LinearRegressionLatLong', 'DecisionTreeLatLong', 'RandomForestLatLong']:
            # These models use only latitude and longitude
            speed_prediction = model.predict([[latitude, longitude]])
            print(latitude, longitude)
        elif selected_model in ['LinearRegressionTSLatLong', 'DecisionTreeTSLatLong']:
            # These models use timestamp, latitude, and longitude
            speed_prediction = model.predict([[timestamp, latitude, longitude]])
            print(timestamp, latitude, longitude)
        elif selected_model in ['LinearRegressionTSIRHLatLong', 'DecisionTreeTSIRHLatLong', 'RandomForestTSIRHLatLong']:
            # These models use timestamp, latitude, and longitude
            speed_prediction = model.predict([[timestamp, intersection, latitude, longitude, road_hierarchy]])
            print(timestamp, intersection, latitude, longitude, road_hierarchy)
        else:
            raise ValueError('Invalid model selection')

        response = {
            'success': True,
            'prediction': speed_prediction.tolist()[0]
        }
    except Exception as e:
        response = {
            'success': False,
            'error': str(e)
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
