from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    features = [float(request.form[field]) for field in request.form]

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
