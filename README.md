# 🩺 Multiple Disease Prediction

## ✨ Project Overview

This project aims to build a **scalable and accurate system** for multi-disease prediction, leveraging machine learning to assist in early detection, improve healthcare provider decision-making, and reduce diagnostic time and cost. The system offers predictions for Kidney disease, Liver disease, and Parkinson's disease through a user-friendly web interface.

## 🏛️ System Architecture

The dashboard follows a clear architecture:

1.  **Frontend:** The user interface for data input (symptoms, test results) is built using **Streamlit**.
2.  **Backend:** Handles user inputs and interacts with the prediction models, developed using **Python**.
3.  **Machine Learning Models:** Predictive models that process input data and provide probabilities for each disease. Algorithms used include **XGBoost** (for Liver disease) and **Random Forest** (for Kidney and Parkinson's disease).

## 💡 Key Features

* **Multi-disease Prediction:** Predicts the likelihood of Kidney disease, Liver disease, and Parkinson's disease.
* **User-friendly Interface:** Simplified input forms and clear prediction results for ease of use.
* **Secure Data Handling:** Ensures user privacy and compliance with data protection regulations.
* **Scalable System:** Designed to support a large number of users and diseases.

## 🔄 Workflow

1.  **Input Data:** Users enter symptoms, demographic details, and test results into the Streamlit interface.
2.  **Data Preprocessing:** The system handles missing values, encodes categorical data, and scales numerical features.
3.  **Model Inference:** Trained predictive models process the preprocessed input data.
4.  **Output:** Displays predicted diseases with their respective probabilities and risk levels to the user.

## 🛠️ Implementation Details

### Data Collection
The models were trained using datasets for:
* Parkinson's
* Kidney Disease
* Indian Liver Patient

### Data Preprocessing
Key steps included:
* Handling missing data (e.g., imputation).
* Encoding categorical variables.
* Feature scaling (e.g., SMOTE, StandardScaler).

### Model Training
Separate models were trained for each disease, with cross-validation employed to ensure robustness.

### Model Evaluation
Performance was assessed using metrics such as:
* **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.

## 💻 Tools and Technologies

* **Programming Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Joblib
* **Frontend Framework:** Streamlit

## ⚙️ Setup Instructions

### Installation

1.  **Install the required Python libraries:**
    ```bash
    pip install streamlit scikit-learn joblib pandas numpy
    ```

2.  **Add the trained model files:**
    Ensure the following `.pkl` files (which store the trained models, scalers, and feature columns) are placed in the same directory as your `app.py` script. These files are essential for the application's functionality.

    * `best_xgboost_model_liver.pkl`
    * `scaler_liver.pkl`
    * `liver_feature_columns.pkl`
    * `tuned_random_forest_ckd.pkl`
    * `scaler_ckd.pkl`
    * `ckd_feature_columns.pkl`
    * `best_random_forest_model_parkinsons.pkl`
    * `scaler_parkinsons.pkl`
    * `parkinsons_feature_columns.pkl`

## 🚀 How to Run the Application

After setting up your environment and placing the necessary model files, you can launch the Streamlit dashboard from your terminal:

```bash
streamlit run MDP.py
