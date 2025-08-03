from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

# Load the trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     return "Diabetes Expert System API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input
    if not data or 'BMI' not in data or 'BloodPressure' not in data:
        return jsonify({'error': 'Missing BMI or BloodPressure values.'}), 400

    # Extract and prepare inputs
    bmi = float(data['BMI'])
    bp = float(data['BloodPressure'])
    input_values = [bmi, bp]
    prediction = model.predict([input_values])[0]

    # Coefficients and intercept
    coefficients = dict(zip(['BMI', 'BloodPressure'], model.coef_))
    intercept = model.intercept_

    # Contributions to prediction
    bmi_contrib = bmi * coefficients['BMI']
    bp_contrib = bp * coefficients['BloodPressure']

    # Risk level
    if prediction > 125:
        risk = "High"
        risk_msg = "This suggests you may be at high risk for diabetes. It's advisable to consult a healthcare provider."
    elif prediction > 100:
        risk = "Moderate"
        risk_msg = "This suggests a moderate risk. Consider making lifestyle adjustments and monitoring your health regularly."
    else:
        risk = "Low"
        risk_msg = "This indicates a low risk. Keep maintaining a healthy lifestyle."

    # Natural language explanation
    explanation = (
        f"Based on your BMI of {bmi} and blood pressure of {bp}, "
        f"the model estimated a diabetes score of {round(prediction, 2)}. "
        f"BMI contributed {round(bmi_contrib, 2)} points, and blood pressure contributed {round(bp_contrib, 2)} points "
        f"to this score, with a base value of {round(intercept, 2)}. "
        f"Your estimated risk level is '{risk}'. {risk_msg}"
    )

    return jsonify({
        "coefficients": {k: round(v, 2) for k, v in coefficients.items()},
        "contributions": {
            "BMI": round(bmi_contrib, 2),
            "BloodPressure": round(bp_contrib, 2)
        },
        "intercept": round(intercept, 2),
        "prediction": round(prediction, 2),
        "risk": risk,
        "explanation": explanation
    })


# if __name__ == '__main__':
#     app.run(debug=True) #removed debug=True for deployment
