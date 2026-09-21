import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🍄 Mushroom Classification")
st.write(
    "Predict whether a mushroom is edible or poisonous "
    "using a Machine Learning model."
)

st.info(
    "⚠️ This application is for educational purposes only. "
    "Do not use the prediction to decide whether a real mushroom is safe to eat."
)

# --------------------------------------------------
# Load Model and Preprocessing Files
# --------------------------------------------------

@st.cache_resource
def load_files():

    model = pickle.load(
        open("archive (3)/mushroom_model_rf.pkl", "rb")
    )

    scaler = pickle.load(
        open("archive (3)/mushroom_scaler.pkl", "rb")
    )

    label_encoder = pickle.load(
        open("archive (3)/mushroom_label_encoder.pkl", "rb")
    )

    data = pd.read_csv(
        "archive (3)/mushrooms.csv"
    )

    return model, scaler, label_encoder, data


model, scaler, label_encoder, data = load_files()

# --------------------------------------------------
# Prepare Feature Information
# --------------------------------------------------

target_column = "class"

X = data.drop(columns=[target_column])

# Create the same dummy variables used during training
X_encoded = pd.get_dummies(X)

# Get the feature names expected by the trained model
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
else:
    model_features = list(X_encoded.columns)

# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Enter Mushroom Characteristics")

input_data = {}

# Create input boxes for every original feature
for column in X.columns:

    values = sorted(X[column].dropna().unique().tolist())

    input_data[column] = st.selectbox(
        column.replace("_", " ").title(),
        values
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Mushroom", type="primary"):

    # Convert user input into DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Make sure input has exactly the same columns as training data
    input_encoded = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    # Apply the saved scaler
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)

    # Convert encoded prediction back to original label
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if str(predicted_class).lower() in ["e", "edible"]:

        st.success("🍄 Prediction: EDIBLE")

    else:

        st.error("☠️ Prediction: POISONOUS")

    st.write("Predicted class:", predicted_class)