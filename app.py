
import os
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calories Predictor")
st.write(
    "Enter the beverage information to estimate its calories."
)

# ---------------- MODEL PATH ----------------
MODEL_PATH = os.path.join(
    "/kaggle/working",
    "starbucks_calorie_model.pkl"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_package(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found! Make sure "
        "'starbucks_calorie_model.pkl' exists in /kaggle/working."
    )
    st.stop()

try:
    package = load_package(MODEL_PATH)

    model = package["model"]
    features = package["features"]

    # Training data is used to create suitable input fields.
    training_data = package.get("training_data")

    if training_data is None:
        st.error(
            "Training data is missing from the saved model package. "
            "Retrain and save the model using the corrected training code."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

# ---------------- INPUT FORM ----------------
st.subheader("Beverage Information")

user_values = {}

with st.form("prediction_form"):

    for feature in features:

        column = training_data[feature]

        # Numeric input
        if pd.api.types.is_numeric_dtype(column):

            numeric_column = pd.to_numeric(
                column, errors="coerce"
            )

            default_value = numeric_column.median()

            if pd.isna(default_value):
                default_value = 0.0

            # Avoid negative values for nutrition inputs.
            minimum_value = 0.0

            user_values[feature] = st.number_input(
                label=str(feature),
                min_value=minimum_value,
                value=max(minimum_value, float(default_value)),
                step=1.0,
                key=f"input_{feature}"
            )

        # Categorical input
        else:
            options = (
                column.dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if options:
                user_values[feature] = st.selectbox(
                    label=str(feature),
                    options=options,
                    key=f"input_{feature}"
                )
            else:
                user_values[feature] = ""

    submitted = st.form_submit_button("Predict Calories")

# ---------------- PREDICTION ----------------
if submitted:

    try:
        input_df = pd.DataFrame(
            [user_values],
            columns=features
        )

        # Keep categorical columns as strings where appropriate.
        # Convert numeric columns to numeric values.
        for feature in features:
            if pd.api.types.is_numeric_dtype(
                training_data[feature]
            ):
                input_df[feature] = pd.to_numeric(
                    input_df[feature],
                    errors="coerce"
                )

        prediction = float(model.predict(input_df)[0])

        st.success(
            f"Estimated Calories: {prediction:.0f} kcal"
        )

        st.caption(
            "This is a machine-learning estimate, "
            "not an official Starbucks nutrition value."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Starbucks Calories Prediction | Machine Learning Project"
)
