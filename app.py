from flask import Flask, request, render_template
import requests
import json
import pandas as pd
import numpy as np
import pickle
import joblib

app = Flask(__name__)

# load the machine learning models from the pickle and joblib files
model_files = {
    'Decision Tree': {
        'model_path': 'decision_tree_model.pkl',
        'scaler_path': 'decision_tree_model.joblib'
    },
    'Linear Regression': {
        'model_path': 'linear_regression_model.pkl',
        'scaler_path': 'linear_regression_model.joblib'
    },
    'Random Forest': {
        'model_path': 'random_forest_model.pkl',
        'scaler_path': 'random_forest_model.joblib'
    }
}

models = {}
scalers = {}
for model_name, files in model_files.items():
    with open(files['model_path'], 'rb') as f:
        model = pickle.load(f)
        models[model_name] = model

    scaler = joblib.load(files['scaler_path'])
    scalers[model_name] = scaler


@app.route('/')
def home():
    return render_template('home.html', models=models.keys())


@app.route('/predict', methods=['POST'])
def predict_view():
    # get the input values from the HTML form
    model_name = request.form['model']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    open_price = float(request.form['open_price'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])
    volume = int(request.form['volume'])

    # fetch the historical bitcoin prices from the API based on the start and end dates
    url = 'https://api.coindesk.com/v1/bpi/historical/close.json?start={start_date}&end={end_date}'
    response = requests.get(url)
    response_json = json.loads(response.text)
    bpi_data = pd.DataFrame(response_json['bpi'].items(), columns=['Date','Close'])
   # bpi_data = pd.DataFrame(response_json['bpi'].items(), columns=['Date','Close'])
    bpi_data['Date'] = pd.to_datetime(bpi_data['Date'])
    bpi_data = bpi_data.set_index('Date')

    # preprocess the input values using the selected scaler
    input_data = [[open_price, high_price, low_price, volume]]
    scaler = scalers[model_name]
    scaled_input = scaler.transform(input_data)

    # make a prediction using the selected model
    model = models[model_name]
    prediction = model.predict(scaled_input)

    # calculate the actual bitcoin price change during the given time period
    actual_price_change = (bpi_data.iloc[-1]['Close'] - bpi_data.iloc[0]['Close']) / bpi_data.iloc[0]['Close'] * 100

    # return the result to the HTML template
    return render_template('result.html', prediction=prediction, actual_price_change=actual_price_change)


if __name__ == '__main__':
    app.run(debug=True)
