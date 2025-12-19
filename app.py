import gradio as gr
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("XGB_heart_disease_predictor_model_1.joblib")

# Class explanations
CLASS_EXPLANATION = {
    0: "✅ No heart disease detected. The patient shows a low likelihood of cardiovascular disease.",
    1: "🟡 Mild heart disease. Early signs detected; lifestyle changes and routine monitoring are advised.",
    2: "🟠 Moderate heart disease. Medical intervention and closer supervision are recommended.",
    3: "🔴 Severe heart disease. High cardiovascular risk; urgent medical evaluation is required.",
    4: "🚨 Critical heart disease. Very high risk of cardiac events; immediate specialist care is essential."
}

def predict_heart_disease(
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    pred = int(model.predict(input_df)[0])
    explanation = CLASS_EXPLANATION[pred]

    return (
        f" Heart Disease Severity Class: {pred}",
        explanation
    )

# ---------------- UI ---------------- #

with gr.Blocks(theme=gr.themes.Soft()) as app:

    # Full-screen gradient background
    gr.HTML("""
    <div style="
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        z-index: -1;
    "></div>
    """)

    # Hero Section
    gr.Markdown("""
    <div style="text-align:center; padding:30px; color:white;">
        <h1 style="text-shadow: 0 0 20px #00ffff;">Heart Disease Prediction System</h1>
        <h3 style="text-shadow: 0 0 10px #00ffff;">AI-powered cardiovascular risk assessment</h3>
        <p><b>Prediction Classes:</b> 0 (No disease) → 4 (Critical disease)</p>
    </div>
    """)

    gr.Markdown("### 📋 Patient Clinical Information", elem_classes=["gr-box"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🧍 Demographics & Vital Signs")
            age = gr.Number(label="Age (years)", value=55)
            sex = gr.Number(label="Sex (0 = Female, 1 = Male)", value=1)
            trestbps = gr.Number(label="Resting Blood Pressure (mmHg)", value=120)
            chol = gr.Number(label="Serum Cholesterol (mg/dL)", value=200)
            thalach = gr.Number(label="Maximum Heart Rate Achieved", value=150)

        with gr.Column():
            gr.Markdown("#### ❤️ Cardiac Risk Indicators")
            cp = gr.Number(
                label="Chest Pain Type (0–3)",
                value=0,
                info="0: Typical angina | 1: Atypical | 2: Non-anginal | 3: Asymptomatic"
            )
            fbs = gr.Number(label="Fasting Blood Sugar (0 ≤120, 1 >120)", value=0)
            restecg = gr.Number(
                label="Resting ECG (0–2)",
                value=0,
                info="0: Normal | 1: ST-T abnormality | 2: LV hypertrophy"
            )
            exang = gr.Number(label="Exercise-Induced Angina (0 = No, 1 = Yes)", value=0)
            oldpeak = gr.Number(label="ST Depression (Oldpeak)", value=1.0)

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🧪 Advanced Diagnostic Parameters")
            slope = gr.Number(label="Slope of ST Segment (0–2)", value=1)
            ca = gr.Number(label="Number of Major Vessels (0–4)", value=0)
            thal = gr.Number(
                label="Thalassemia (0–3)",
                value=1,
                info="0: Normal | 1: Fixed defect | 2: Reversible defect"
            )

    gr.Markdown("---")

    predict_btn = gr.Button(
        "🔍 Predict Heart Disease Risk",
        variant="primary",
        size="lg"
    )

    with gr.Row():
        output_class = gr.Textbox(
            label="🩺 Prediction Result",
            interactive=False
        )
        output_explanation = gr.Textbox(
            label="📌 Clinical Interpretation",
            interactive=False
        )

    predict_btn.click(
        predict_heart_disease,
        inputs=[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ],
        outputs=[output_class, output_explanation]
    )

    gr.Markdown("""
    ---
    ⚠️ **Medical Disclaimer**  
    This application is intended **strictly for educational and research purposes**.  
    It does **not** replace professional medical diagnosis or treatment.
    """, elem_classes=["gr-box"])

app.launch()
