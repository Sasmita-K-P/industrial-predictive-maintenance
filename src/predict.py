import joblib
import pandas as pd

model = joblib.load("models/xgb_model.pkl")

new_machine = pd.DataFrame({
    "Type":[2],
    "Air_temperature_K":[300],
    "Process_temperature_K":[310],
    "Rotational_speed_rpm":[1400],
    "Torque_Nm":[55],
    "Tool_wear_min":[220]
})

prediction = model.predict(new_machine)

if prediction[0] == 1:
    print("⚠ Machine Failure Likely")
else:
    print("✅ Machine Operating Normally")