import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Load class names
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# Dictionary: Treatment advice for all classes (default)
treatment_advice = {
    name: "🛠️ No specific advice available. Please consult a local expert or extension officer."
    for name in class_names
}

# Custom advice for known diseases
treatment_advice.update({
    "Pepper__bell___Bacterial_spot": (
        "Remove infected leaves. Apply a copper-based fungicide weekly during wet conditions. "
        "Ensure good airflow between plants. Use disease-free seeds and avoid overhead irrigation."
    ),
    "Pepper__bell___healthy": (
        "No disease symptoms detected. Maintain consistent watering, mulching, and balanced nutrition to prevent stress."
    ),

    "Potato___Early_blight": (
        "Start preventive fungicide sprays (chlorothalonil, mancozeb) early in the season. "
        "Rotate crops annually. Avoid watering late in the day to reduce humidity."
    ),
    "Potato___Late_blight": (
        "Apply systemic fungicides like metalaxyl or cymoxanil immediately. "
        "Destroy infected foliage. Avoid planting potatoes near tomatoes and space plants properly."
    ),
    "Potato___healthy": (
        "Plant appears healthy. Practice crop rotation and maintain proper soil drainage to reduce future disease risk."
    ),

    "Tomato_Bacterial_spot": (
        "Avoid working with wet plants. Apply fixed copper sprays with mancozeb. "
        "Use resistant varieties and disinfect tools between uses."
    ),
    "Tomato_Early_blight": (
        "Remove lower, infected leaves. Apply fungicides every 7–10 days. "
        "Stake or cage plants to improve air circulation and reduce splash spread."
    ),
    "Tomato_Late_blight": (
        "Apply protectant fungicides (chlorothalonil) before symptoms appear and systemic ones (like fluopicolide) after detection. "
        "Remove infected plants immediately. Avoid overhead irrigation."
    ),
    "Tomato_Leaf_Mold": (
        "Increase ventilation in greenhouses or dense plantings. Apply sulfur-based fungicides. "
        "Water at the base of the plant, preferably in the morning."
    ),
    "Tomato_Septoria_leaf_spot": (
        "Prune affected foliage. Apply fungicides containing copper or mancozeb. "
        "Avoid wetting leaves. Rotate crops annually to non-solanaceous plants."
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Spray with neem oil or insecticidal soap. Introduce natural predators (e.g., ladybugs, predatory mites). "
        "Avoid dusty conditions and over-fertilization."
    ),
    "Tomato__Target_Spot": (
        "Remove infected leaves promptly. Apply strobilurin-class fungicides. "
        "Keep foliage dry and ensure proper row spacing."
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "Remove and destroy infected plants. Control whiteflies using yellow sticky traps and systemic insecticides. "
        "Avoid planting susceptible varieties in high-risk areas."
    ),
    "Tomato__Tomato_mosaic_virus": (
        "Do not smoke around plants. Disinfect hands and tools. Remove and destroy symptomatic plants. "
        "Avoid saving seeds from infected crops."
    ),
    "Tomato_healthy": (
        "No symptoms of disease. Continue regular care: water consistently, apply mulch, and monitor weekly for any changes."
    )

})

# Image preprocessing
def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Estimate severity based on top-1 confidence
def get_severity(confidence):
    if confidence > 0.9:
        return "Severe"
    elif confidence > 0.7:
        return "Moderate"
    else:
        return "Mild"

# Predict top 3 diseases
def predict_top3(image):
    processed = preprocess(image)
    probs = model.predict(processed)[0]
    top_indices = probs.argsort()[-3:][::-1]
    results = [(class_names[i], float(probs[i])) for i in top_indices]
    return results

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🩺 GreenCheck")
st.markdown("Upload one or more leaf images to get predictions, severity, and treatment advice.")

# Image upload
uploaded_files = st.file_uploader("Upload leaf image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        st.subheader(f"📷 {uploaded_file.name}")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        top_preds = predict_top3(image)
        labels = [label for label, _ in top_preds]
        scores = [score for _, score in top_preds]

        # Display bar chart
        st.bar_chart(pd.DataFrame({"Probability": scores}, index=labels))

        # Show top prediction details
        top_label, confidence = top_preds[0]
        severity = get_severity(confidence)
        advice = treatment_advice.get(top_label)

        st.markdown(f"**Top Prediction:** {top_label} ({confidence*100:.2f}%)")
        st.markdown(f"**Severity Level:** {severity}")
        st.markdown("📋 **Recommended Action:**")
        st.info(advice)
        st.markdown("---")

        # Collect results for CSV
        results.append({
            "Filename": uploaded_file.name,
            "Top Prediction": top_label,
            "Confidence (%)": f"{confidence*100:.2f}",
            "Severity": severity,
            "Advice": advice,
            "Top 2": labels[1],
            "Top 2 Confidence (%)": f"{scores[1]*100:.2f}",
            "Top 3": labels[2],
            "Top 3 Confidence (%)": f"{scores[2]*100:.2f}"
        })

    # Offer CSV download
    if results:
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="plant_disease_predictions.csv",
            mime="text/csv"
        )
