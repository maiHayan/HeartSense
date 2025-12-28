import numpy as np
import gradio as gr
import pickle

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# Prediction Function
def predict_heart(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Validation
    if age <= 0 or trestbps <= 0 or chol <= 0 or thalach <= 0:
        return "<div style='color:orange; font-size:20px; font-weight:bold'>⚠️ Invalid Input – Enter realistic values (Age 30–90)</div>"
    
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        return "<h2 style='color:red'>🔴 HIGH RISK – Heart Disease Likely!</h2>" \
               "<p>👉 Consult a cardiologist. Low cholesterol diet, avoid smoking, BP control, reduce stress, ECG/TMT/ECHO recommended.</p>"
    else:
        return "<h2 style='color:green'>🟢 SAFE – No Heart Disease Detected</h2>" \
               "<p>💡 Maintain healthy lifestyle & routine checkups.</p>"

# Inputs
inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Sex (1 = Male, 0 = Female)"),
    gr.Number(label="Chest Pain Type"),
    gr.Number(label="Resting BP"),
    gr.Number(label="Cholesterol"),
    gr.Number(label="Fasting Sugar"),
    gr.Number(label="Resting ECG"),
    gr.Number(label="Max Heart Rate"),
    gr.Number(label="Exercise Angina"),
    gr.Number(label="Oldpeak"),
    gr.Number(label="Slope"),
    gr.Number(label="CA"),
    gr.Number(label="Thal")
]

# UI
app = gr.Interface(
    fn=predict_heart,
    inputs=inputs,
    outputs=gr.HTML(label="Result"),
    title="Heart Disease Prediction - HeartSense",
    description="Enter data to check risk level & recommendations",
    allow_flagging=False
)

# Launch
if __name__ == "__main__":
    app.launch()
