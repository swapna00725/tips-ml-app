import pandas as pd
import joblib

import pickle


# 📦 Load the saved model
#loaded_model = joblib.load('tips_model.joblib')

# 📦 Load the saved .pkl model
with open('tips_modelpkle.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

# ✨ Create manual input data for prediction
test_data1= pd.DataFrame({
    'total_bill': [10.9],
    'tip': [1.98],
    'sex':['Female'],
    'smoker': ['No'],
    'day':['Sun'],
    'size':[3]
})

# 🔮 Predict using the loaded model
prediction= model_loaded.predict(test_data1)

# 🎯 Output results
#for i, pred in enumerate(predictions):
    #print(f"Passenger {i+1} ➜ {'Survived 🛟' if pred == 1 else 'Did NOT Survive 💀'}")
print(f"prediction from loaded model {prediction}")