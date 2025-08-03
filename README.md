# Diabetes Expert System: A Hybrid Machine Learning and Rule-Based API

This project demonstrates a **hybrid expert system** for predicting diabetes risk using a machine learning model (Linear Regression) and rule-based logic to categorize risk levels. It consists of two main components:
1. **Training Code**: Trains a linear regression model using the Pima Indians Diabetes dataset to predict a glucose-based diabetes score from BMI and Blood Pressure.
2. **Flask API**: Provides a web interface to input BMI and Blood Pressure, make predictions, and explain the results with risk levels and advice.

This system is a great learning tool for understanding machine learning, APIs, and how to combine data-driven predictions with rule-based decision-making.

---

## **Learning Objectives**
- Understand how to train a linear regression model using scikit-learn.
- Learn how to save and load a trained model with joblib.
- Explore how to build a web API using Flask to serve machine learning predictions.
- See how a hybrid system combines machine learning (predictions) with rule-based logic (risk levels and explanations).
- Interpret model outputs like coefficients, intercept, and contributions to explain predictions.

---

## **Project Overview**
The system predicts a diabetes risk score based on two inputs: **BMI** (Body Mass Index) and **Blood Pressure**. The machine learning model generates a numerical score, which is then categorized into risk levels ("Low," "Moderate," or "High") using predefined rules. The API provides a detailed explanation of how the inputs contribute to the prediction, making it a hybrid of data-driven and rule-based approaches.

---

## **Prerequisites**
Before running the project, ensure you have:
- **Python 3.8+** installed.
- **pip** (Python package manager).
- A basic understanding of Python, machine learning concepts (e.g., linear regression), and APIs.
- A code editor like VS Code or PyCharm.
- (Optional) Tools like Postman or `curl` to test the API.

---

## **Setup Instructions**

### **1. Clone or Download the Project**
- Create a project folder (e.g., `diabetes-expert-system`).
- Save the two main scripts: `train_model.py` (training code) and `app.py` (Flask API code).

### **2. Install Dependencies**
Install the required Python libraries by running:
```bash
pip install pandas scikit-learn joblib flask
```

### **3. Files in the Project**
- `train_model.py`: Trains the linear regression model and saves it as `diabetes_model.pkl`.
- `app.py`: Runs the Flask API to make predictions and provide explanations.
- `diabetes_model.pkl`: The trained model file (generated after running `train_model.py`).
- `README.md`: This file.

---

## **How It Works**

### **Step 1: Training the Model (`train_model.py`)**
This script trains a linear regression model to predict a diabetes score (based on glucose levels) using BMI and Blood Pressure.

**Key Steps**:
1. **Load Data**: Uses the Pima Indians Diabetes dataset from a GitHub URL.
2. **Select Features**: Uses BMI and Blood Pressure as input features and Glucose as the target.
3. **Train-Test Split**: Splits data into 80% training and 20% testing sets.
4. **Train Model**: Fits a linear regression model to learn coefficients (weights for BMI and Blood Pressure) and an intercept (base value).
5. **Save Model**: Saves the trained model to `diabetes_model.pkl`.

**Code**:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Select features and target
X = df[['BMI', 'BloodPressure']]
y = df['Glucose']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'diabetes_model.pkl')
```

**How to Run**:
1. Save the code as `train_model.py`.
2. Run it:
   ```bash
   python train_model.py
   ```
3. This creates `diabetes_model.pkl` in the project folder.

**Learning Tip**: After training, you can print the model’s coefficients and intercept to understand their values:
```python
print("Coefficients (BMI, BloodPressure):", model.coef_)
print("Intercept:", model.intercept_)
```

---

### **Step 2: Running the Flask API (`app.py`)**
This script creates a web API that uses the trained model to predict diabetes risk and explain the results.

**Key Features**:
- **Homepage (`/`)**: Confirms the API is running.
- **Prediction Endpoint (`/predict`)**: Accepts POST requests with BMI and Blood Pressure, returns a prediction, risk level, and explanation.
- **Hybrid Approach**:
  - **Machine Learning**: Uses the linear regression model to predict a glucose-based diabetes score.
  - **Rule-Based**: Categorizes the score into "Low" (≤100), "Moderate" (101–125), or "High" (>125) risk levels with tailored advice.

**Code**:
```python
from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('diabetes_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return "Diabetes Expert System API is running."

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

if __name__ == '__main__':
    app.run(debug=True)
```

**How to Run**:
1. Ensure `diabetes_model.pkl` is in the same folder (generated by `train_model.py`).
2. Save the code as `app.py`.
3. Run it:
   ```bash
   python app.py
   ```
4. The API will start at `http://localhost:5000`.

---

### **Using the API**
1. **Check the Homepage**:
   - Open a browser and visit `http://localhost:5000/`.
   - You should see: "Diabetes Expert System API is running."

2. **Make a Prediction**:
   - Send a POST request to `http://localhost:5000/predict` with JSON data containing BMI and Blood Pressure.
   - Example using `curl`:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"BMI": 25, "BloodPressure": 120}' http://localhost:5000/predict
     ```
   - Example using Python:
     ```python
     import requests
     data = {"BMI": 25, "BloodPressure": 120}
     response = requests.post("http://localhost:5000/predict", json=data)
     print(response.json())
     ```
   - Example response:
     ```json
     {
       "coefficients": {"BMI": 0.12, "BloodPressure": 0.79},
       "contributions": {"BMI": 3.0, "BloodPressure": 94.8},
       "intercept": 50.35,
       "prediction": 148.15,
       "risk": "High",
       "explanation": "Based on your BMI of 25 and blood pressure of 120, the model estimated a diabetes score of 148.15. BMI contributed 3.0 points, and blood pressure contributed 94.8 points to this score, with a base value of 50.35. Your estimated risk level is 'High'. This suggests you may be at high risk for diabetes. It's advisable to consult a healthcare provider."
     }
     ```

---

### **Understanding the Hybrid Expert System**
This system is "hybrid" because it combines:
1. **Machine Learning**:
   - The linear regression model predicts a numerical diabetes score (glucose level) based on BMI and Blood Pressure.
   - The model learns **coefficients** (weights for each feature) and an **intercept** (base value) during training.
   - Formula: `Score = intercept + (BMI × BMI_coefficient) + (BloodPressure × BloodPressure_coefficient)`.

2. **Rule-Based Logic**:
   - The API uses predefined rules to categorize the score:
     - **High Risk**: Score > 125
     - **Moderate Risk**: 101 ≤ Score ≤ 125
     - **Low Risk**: Score ≤ 100
   - Each risk level includes a tailored message with health advice.

3. **Explanations**:
   - The API calculates **contributions** (e.g., BMI × BMI_coefficient) to show how each input affects the score.
   - It provides a natural language explanation to make the prediction understandable.

**Example**:
- Input: BMI = 25, Blood Pressure = 120.
- Model: Coefficients = [0.123456, 0.789012], Intercept = 50.345678.
- Contributions: BMI = 25 × 0.123456 = 3.0864, Blood Pressure = 120 × 0.789012 = 94.68144.
- Prediction: 50.345678 + 3.0864 + 94.68144 = 148.113518.
- Risk: High (since 148.11 > 125).
- Explanation: Combines all details into a readable summary.

---

### **Learning Exercises**
1. **Explore the Model**:
   - Modify `train_model.py` to print the coefficients and intercept after training.
   - Experiment with different features from the dataset (e.g., add `Age` or `Insulin`) and retrain the model.

2. **Test the API**:
   - Try different BMI and Blood Pressure values to see how the risk level changes.
   - Send invalid inputs (e.g., missing BMI) to test error handling.

3. **Extend the System**:
   - Add more features to the model and update the API to handle them.
   - Modify the risk thresholds (e.g., change 125 to 130) or add new risk categories.
   - Improve input validation (e.g., check for negative BMI).

4. **Understand Coefficients**:
   - Calculate contributions manually for a given input and compare with the API’s output.
   - Discuss why certain features (e.g., Blood Pressure) might have a higher coefficient than others.

---

### **Limitations**
- The model uses only BMI and Blood Pressure, which oversimplifies diabetes risk.
- The risk thresholds (100, 125) are arbitrary and for demonstration only. Real medical systems use more complex criteria.
- The API lacks advanced input validation (e.g., negative values are accepted).
- This is an educational tool, not a substitute for professional medical advice.

---

### **For Instructors**
- **Teaching Tips**:
  - Start with the training code to explain linear regression and model training.
  - Use the API to demonstrate real-world applications of machine learning.
  - Discuss the hybrid nature: how machine learning (predictions) and rules (risk levels) work together.
- **Assignments**:
  - Ask students to add new features or improve the API’s user interface.
  - Have them analyze the dataset to understand its features and limitations.
  - Challenge them to deploy the API on a cloud platform like Heroku or Render.

---

### **Troubleshooting**
- **Missing `diabetes_model.pkl`**: Run `train_model.py` first to generate the model file.
- **Module Not Found**: Ensure all dependencies (`pandas`, `scikit-learn`, `joblib`, `flask`) are installed.
- **API Not Responding**: Check that the Flask server is running and the URL is correct (`http://localhost:5000`).
- **Invalid JSON**: Ensure POST requests have valid JSON with `BMI` and `BloodPressure` keys.

---

### **Further Reading**
- **Linear Regression**: [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- **Flask APIs**: [Flask Documentation](https://flask.palletsprojects.com/)
- **Pima Indians Diabetes Dataset**: [Kaggle Description](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Hybrid Expert Systems**: Research how machine learning and rule-based systems are combined in real-world applications.

---

This project is an engaging way to learn about machine learning, APIs, and hybrid systems. Have fun exploring, and consult a healthcare professional for real diabetes risk assessments!