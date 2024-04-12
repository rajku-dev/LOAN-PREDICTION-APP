from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a function to convert categorical inputs to one-hot encoded format
def process_input(data):
    # Define mappings for categorical variables
    married_mapping = {'Yes': 1, 'No': 0}
    dependents_mapping = {'1': [1, 0, 0], '2': [0, 1, 0], '3+': [0, 0, 1]}
    gender_mapping={'Male':1,'Female':0}
    education_mapping = {'Yes': 1, 'No': 0}
    self_employed_mapping = {'Yes': 1, 'No': 0}
    property_area_mapping = {'Urban': [1, 0], 'Semiurban': [0, 1]}

    # Convert categorical variables to one-hot encoded format
    processed_data = [
        float(data['ApplicantIncome']),
        float(data['CoapplicantIncome']),
        float(data['LoanAmount']),
        float(data['Loan_Amount_Term']),
        float(data['CreditHistory']),
        gender_mapping[data['Gender']],
        married_mapping[data['Married']],
        *dependents_mapping[data['Dependents']],
        education_mapping[data['Education']],
        self_employed_mapping[data['Self_Employed']],
        *property_area_mapping[data['Property_Area']]
    ]

    return np.array(processed_data)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    features = process_input(request.form)

    # Make prediction
    prediction = model.predict([features])

    # Display prediction result
    if prediction[0] == 1:
        result = 'Approved✅'
    else:
        result = 'Not Approved❌'

    # Pass the result to the result.html template
    return render_template('result.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
