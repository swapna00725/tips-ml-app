from flask import Flask, render_template, request
import pandas as pd
import joblib

# 🎉 Load the saved model
model1 = joblib.load('tips_model1.joblib')

# ⚙️ Create Flask app
app = Flask(__name__)

# 🏠 Home route - form page
@app.route('/')
def home():
    return render_template('index.html')

# 🚀 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    to = float(request.form['total_bill'])
    ti = float(request.form['tip'])
    se = request.form['sex']
    sm = request.form['smoker']
    da = request.form['day']
    si = int(request.form['size'])

    # 👩‍🔬 Convert input into DataFrame
 
    input_data= pd.DataFrame({
    'total_bill': [to],
    'tip': [ti],
    'sex':[se],
    'smoker': [sm],
    'day':[da],
    'size':[si]
})

    # 🔮 Make prediction
    pred = model1.predict(input_data)[0]
    result = "dinner 🛟" if pred == 0 else "lunch"

    return render_template('index.html', prediction=result)

# 🌍 Run the app
if __name__ == '__main__':
    app.run(debug=True)