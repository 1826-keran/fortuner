from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained Oil Price model
model = joblib.load('uso_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        spx = float(request.form['spx'])
        gold = float(request.form['gold'])
        slv = float(request.form['slv'])
        eurusd = float(request.form['eurusd'])

        features = np.array([[spx, gold, slv, eurusd]])
        prediction = model.predict(features)
        return render_template('index.html',
                               prediction_text=f'Predicted Oil Price: {prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
