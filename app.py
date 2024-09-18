from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    rfc = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    ms = pickle.load(scaler_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            N_SOIL = float(request.form['N_SOIL'])
            P_SOIL = float(request.form['P_SOIL'])
            K_SOIL = float(request.form['K_SOIL'])
            TEMPERATURE = float(request.form['TEMPERATURE'])
            HUMIDITY = float(request.form['HUMIDITY'])
            ph = float(request.form['ph'])
            RAINFALL = float(request.form['RAINFALL'])

            # Check if the inputs are within a valid range
            if (N_SOIL < 0 or P_SOIL < 0 or K_SOIL < 0 or 
                TEMPERATURE < 0 or HUMIDITY < 0 or ph < 0 or RAINFALL < 0):
                return redirect(url_for('recommendation', result="Invalid input. All values should be non-negative."))
            
            # Prepare input for prediction
            predict_input = np.array([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL]])
            predict_input_scaled = ms.transform(predict_input)
            prediction = rfc.predict(predict_input_scaled)[0]
            
            return redirect(url_for('recommendation', result=prediction))
        except ValueError as ve:
            return redirect(url_for('recommendation', result="Invalid input. Please enter a valid number for all fields."))
        except Exception as e:
            return redirect(url_for('recommendation', result="An error occurred: " + str(e)))

    return render_template('index.html')

@app.route('/recommendation')
def recommendation():
    result = request.args.get('result', 'No recommendation available.')
    return render_template('recommendation.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
