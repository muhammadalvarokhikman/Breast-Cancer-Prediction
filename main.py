import streamlit as st
import numpy as np
import joblib

# =======================================================
# Load Model, Scaler, PCA
# =======================================================
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
model = joblib.load("models/model.pkl")

# =======================================================
# Prediction Pipeline
# =======================================================
def predict(data):
    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    return model.predict(data_pca)[0]


# =======================================================
# Streamlit UI Custom Styling
# =======================================================
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.markdown("""
    <h1 style='text-align:center; color:white;'>Breast Cancer Diagnosis Prediction</h1>
    <p style='text-align:center; font-size:18px; color:#cccccc;'>
        Enter the feature values below to predict whether the tumor is <b>Benign</b> or <b>Malignant</b>.
    </p>
""", unsafe_allow_html=True)


# =======================================================
# Feature Groups (Agar Rapi)
# =======================================================
mean_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

se_features = [
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se'
]

worst_features = [
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]


inputs = []


# =======================================================
# Card Container Function
# =======================================================
def feature_card(title, feature_list):
    st.markdown(f"""
        <div style="padding:18px; border-radius:10px; 
             background-color:#1e1e1e; margin-bottom:20px;">
            <h3 style="color:#ffffff;">{title}</h3>
        </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    idx = 0
    for feat in feature_list:
        with cols[idx % 2]:
            val = st.number_input(feat, value=0.0, format="%.6f")
            inputs.append(val)
        idx += 1


# =======================================================
# INPUT SECTIONS (Rapih dan Elegan)
# =======================================================
feature_card("🟦 Mean Features", mean_features)
feature_card("🟨 Standard Error (SE) Features", se_features)
feature_card("🟥 Worst Features", worst_features)

st.markdown("<br>", unsafe_allow_html=True)


# =======================================================
# Prediction Button
# =======================================================
center = st.columns(3)[1]

with center:
    if st.button("🔍 Diagnosis Prediction", use_container_width=True):
        result = predict(inputs)

        if result == 1:
            st.error("**Prediction Result: Malignant**", icon="🚨")
        else:
            st.success("**Prediction Result: Benign**", icon="✅")
