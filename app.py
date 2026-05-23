from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get form values
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Feature array
        features = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        # Prediction
        prediction = model.predict(features)

        # Probability
        probability = model.predict_proba(features)[0][1] * 100

        # Result
        if prediction[0] == 1:
            result = "⚠️ High Risk of Heart Disease"
            status = "danger"
        else:
            result = "✅ Low Risk of Heart Disease"
            status = "success"

        return render_template(
            'index.html',
            prediction_text=result,
            probability=round(probability, 2),
            status=status,
            values=request.form
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {e}",
            status="danger",
            values=request.form
        )


if __name__ == "__main__":
    app.run(debug=True)