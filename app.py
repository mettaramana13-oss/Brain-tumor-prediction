import gradio as gr
import pandas as pd
import joblib
import os

# Load saved model
model = joblib.load("Brain Tumor.pkl")

# Get feature names from trained model
feature_names = list(model.feature_names_in_)

# Prediction function
def predict_brain_tumor(*values):

    new_data = pd.DataFrame(
        [values],
        columns=feature_names
    )

    prediction = model.predict(new_data)

    return str(prediction[0])


# Create Gradio Interface
app = gr.Interface(
    fn=predict_brain_tumor,

    inputs=[
        gr.Number(label=feature)
        for feature in feature_names
    ],

    outputs=gr.Textbox(label="Prediction Result"),

    title="Brain Tumor Classification System",

    description="Enter the 12 feature values to get a model prediction."
)

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
