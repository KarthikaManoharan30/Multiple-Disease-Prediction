import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------------- Load Models and Scalers ----------------------
try:
    liver_model = joblib.load("best_xgboost_model_liver.pkl")
    liver_scaler = joblib.load("scaler_liver.pkl")
    liver_features = joblib.load("liver_feature_columns.pkl")
    
    ckd_model = joblib.load("tuned_random_forest_ckd.pkl")
    ckd_scaler = joblib.load("scaler_ckd.pkl")
    ckd_features = joblib.load("ckd_feature_columns.pkl")
    
    parkinsons_model = joblib.load("best_random_forest_model_parkinsons.pkl")
    parkinsons_scaler = joblib.load("scaler_parkinsons.pkl")
    parkinsons_features = joblib.load("parkinsons_feature_columns.pkl")

except FileNotFoundError as e:
    st.error(f"Required model, scaler, or feature file not found: {e}. Please ensure the files are in the same directory.")
    st.stop()

st.set_page_config(page_title="🩺 Multi-Disease Prediction", layout="wide")
st.title("🩺 Multi-Disease Prediction Dashboard")

# ---------------------- Tabs ----------------------
tab1, tab2, tab3 = st.tabs(["Liver Disease", "Kidney Disease", "Parkinson's Disease"])

# ---------------------- Liver Disease ----------------------
with tab1:
    st.header("🧬 Liver Disease Prediction")
    age = st.number_input('Age', min_value=1, max_value=120, value=45, format="%.4f")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    total_bilirubin = st.number_input('Total Bilirubin', value=1.0000, format="%.4f")
    direct_bilirubin = st.number_input('Direct Bilirubin', value=0.5000, format="%.4f")
    alkphos = st.number_input('Alkaline Phosphotase', value=200.0000, format="%.4f")
    sgpt = st.number_input('Alamine Aminotransferase (SGPT)', value=30.0000, format="%.4f")
    sgot = st.number_input('Aspartate Aminotransferase (SGOT)', value=35.0000, format="%.4f")
    total_proteins = st.number_input('Total Proteins', value=6.5000, format="%.4f")
    albumin = st.number_input('Albumin', value=3.0000, format="%.4f")
    a_g_ratio = st.number_input('Albumin and Globulin Ratio', value=1.0000, format="%.4f")

    gender_encoded = 1 if gender == 'Male' else 0

    if st.button('Predict Liver Disease'):
        liver_data = pd.DataFrame([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                                alkphos, sgpt, sgot, total_proteins, albumin, a_g_ratio]],
                                columns=liver_features)
        scaled_liver_data = liver_scaler.transform(liver_data)
        prediction = liver_model.predict(scaled_liver_data)[0]
        prob = liver_model.predict_proba(scaled_liver_data)[0][1]
        if prediction == 1:
            st.error(f"⚠ Patient likely has liver disease. (Risk score: {prob:.4f})")
        else:
            st.success(f"✅ Patient unlikely to have liver disease. (Risk score: {prob:.4f})")
# ---------------------- Kidney Disease ----------------------
with tab2:
    st.header("🩺 Kidney Disease Prediction")

    # Numeric Inputs
    age = st.number_input('Age ', min_value=1, max_value=120, value=45, format="%.4f", key='kidney_age')
    bp = st.number_input('Blood Pressure', value=80.0000, format="%.4f")
    sg = st.number_input('Specific Gravity (e.g., 1.01)', value=1.0200, format="%.4f")
    al = st.number_input('Albumin', value=1.0000, format="%.4f")
    su = st.number_input('Sugar', value=0.0000, format="%.4f")
    bgr = st.number_input('Blood Glucose Random', value=121.0000, format="%.4f")
    bu = st.number_input('Blood Urea', value=36.0000, format="%.4f")
    sc = st.number_input('Serum Creatinine', value=1.2000, format="%.4f")
    sod = st.number_input('Sodium', value=138.0000, format="%.4f")
    pot = st.number_input('Potassium', value=4.4000, format="%.4f")
    hemo = st.number_input('Hemoglobin', value=15.4000, format="%.4f")
    pcv = st.number_input('Packed Cell Volume', value=44.0000, format="%.4f")
    wc = st.number_input('White Blood Cell Count', value=7800.0000, format="%.4f")
    rc = st.number_input('Red Blood Cell Count', value=5.2000, format="%.4f")

    # Categorical Inputs
    rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
    pc = st.selectbox('Pus Cell', ['normal', 'abnormal'])
    pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
    ba = st.selectbox('Bacteria', ['present', 'notpresent'])
    htn = st.selectbox('Hypertension', ['yes', 'no'])
    dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'])
    cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
    appet = st.selectbox('Appetite', ['good', 'poor'])
    pe = st.selectbox('Pedal Edema', ['yes', 'no'])
    ane = st.selectbox('Anemia', ['yes', 'no'])

    binary_map = {'yes': 1, 'no': 0, 'normal': 1, 'abnormal': 0,
                  'present': 1, 'notpresent': 0, 'good': 1, 'poor': 0}

    if st.button('Predict Kidney Disease'):
        kidney_data = pd.DataFrame([[age, bp, sg, al, su,
                                     binary_map[rbc], binary_map[pc], binary_map[pcc], binary_map[ba],
                                     bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                                     binary_map[htn], binary_map[dm], binary_map[cad], binary_map[appet],
                                     binary_map[pe], binary_map[ane]]], columns=ckd_features)
        try:
            scaled_kidney_data = ckd_scaler.transform(kidney_data)
            prediction = ckd_model.predict(scaled_kidney_data)[0]
            prob = ckd_model.predict_proba(scaled_kidney_data)[0][1]
            if prediction == 1:
                st.error(f"⚠ High risk of CKD. (Risk score: {prob:.4f})")
            else:
                st.success(f"✅ Low risk of CKD. (Risk score: {prob:.4f})")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------- Parkinson's Disease ----------------------
with tab3:
    st.header("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter patient voice measurements below:")

    mdvp_fo = st.number_input('MDVP:Fo(Hz)', value=119.9920, format="%.4f")
    mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', value=157.3020, format="%.4f")
    mdvp_flo = st.number_input('MDVP:Flo(Hz)', value=74.9970, format="%.4f")
    mdvp_jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.0078, format="%.4f")
    mdvp_jitter_abs = st.number_input('MDVP:Jitter(Abs)', value=0.0001, format="%.4f")
    mdvp_rap = st.number_input('MDVP:RAP', value=0.0037, format="%.4f")
    mdvp_ppq = st.number_input('MDVP:PPQ', value=0.0055, format="%.4f")
    jitter_ddp = st.number_input('Jitter:DDP', value=0.0111, format="%.4f")
    mdvp_shimmer = st.number_input('MDVP:Shimmer', value=0.0437, format="%.4f")
    mdvp_shimmer_db = st.number_input('MDVP:Shimmer(dB)', value=0.4260, format="%.4f")
    shimmer_apq3 = st.number_input('Shimmer:APQ3', value=0.0218, format="%.4f")
    shimmer_apq5 = st.number_input('Shimmer:APQ5', value=0.0313, format="%.4f")
    mdvp_apq = st.number_input('MDVP:APQ', value=0.0297, format="%.4f")
    shimmer_dda = st.number_input('Shimmer:DDA', value=0.0654, format="%.4f")
    nhr = st.number_input('NHR', value=0.0221, format="%.4f")
    hnr = st.number_input('HNR', value=21.0330, format="%.4f")
    rpde = st.number_input('RPDE', value=0.4148, format="%.4f")
    dfa = st.number_input('DFA', value=0.8152, format="%.4f")
    spread1 = st.number_input('spread1', value=-4.8130, format="%.4f")
    spread2 = st.number_input('spread2', value=0.2664, format="%.4f")
    d2 = st.number_input('D2', value=2.3014, format="%.4f")
    ppe = st.number_input('PPE', value=0.2846, format="%.4f")

    if st.button("Predict Parkinson's Disease"):
        parkinsons_data = pd.DataFrame([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                                         mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                                         shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
                                         rpde, dfa, spread1, spread2, d2, ppe]],
                                         columns=parkinsons_features)
        scaled_parkinsons_data = parkinsons_scaler.transform(parkinsons_data)
        prediction = parkinsons_model.predict(scaled_parkinsons_data)[0]
        prob = parkinsons_model.predict_proba(scaled_parkinsons_data)[0][1]
        if prediction == 1:
            st.error(f"⚠ Likely Parkinson's Disease. (Risk score: {prob:.4f})")
        else:
            st.success(f"✅ Unlikely to have Parkinson's Disease. (Risk score: {prob:.4f})")