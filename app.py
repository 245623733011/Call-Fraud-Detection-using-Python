from flask import Flask, render_template, request, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

print(app.static_folder)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        duration = float(request.form['duration'])
        call_type = int(request.form['call_type'])
        is_international = int(request.form['is_international'])

        features = np.array([[duration, call_type, is_international]])
        prediction = model.predict(features)[0]

        result = "🚨 Risk: Fraudulent Call Detected" if prediction == 1 else "✅ Safe: Normal Call"
        return render_template('index1.html', prediction_text=result)
    except Exception as e:
        return render_template('index1.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
