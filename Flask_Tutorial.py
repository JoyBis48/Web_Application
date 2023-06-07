import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def predict():
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['bloodPressure'])
    skin_thickness = int(request.form['skinThickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetesPedigreeFunction'])
    age = int(request.form['age'])

    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)
    output = prediction[0]
    if output == 1:
        return render_template('Diabetic_result.html')
    else:
        return render_template('NonDiabetic_result.html')


if __name__ == '__main__':
    app.run(debug=True)
