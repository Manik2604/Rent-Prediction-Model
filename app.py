import os
import gradio as gr
import joblib

# Load the trained model
deployed_lr = joblib.load("Rent_Prediction_Model.pkl")


def predict_rent(size_of_prop):
    try:
        prediction = deployed_lr.predict([[float(size_of_prop)]])
        return f"Estimated Rent: ₹{prediction[0]:,.2f}"
    except Exception as e:
        return f"Error: {e}"


# Custom CSS
custom_css = """
.gradio-container {
    background-image: url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.glass-container {
    background-color: rgba(255,255,255,0.95) !important;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    color: #1f2937 !important;
}

.glass-container h1,
.glass-container h2,
.glass-container h3,
.glass-container p,
.glass-container li,
.glass-container strong,
.glass-container label {
    color: #1f2937 !important;
}

.glass-container a {
    color: #2563eb !important;
    text-decoration: none;
}

.glass-container a:hover {
    text-decoration: underline;
}
"""

# Build Interface
with gr.Blocks(css=custom_css, title="Property Rent Predictor") as interface:

    with gr.Column(elem_classes="glass-container"):

        gr.Markdown(
            "<h1 style='text-align:center;'>🏙️ Property Rent Predictor</h1>"
        )

        gr.Markdown(
            "<p style='text-align:center;'>Enter the property size to estimate the monthly rent using Machine Learning.</p>"
        )

        gr.HTML("<hr>")

        with gr.Row():

            # Left Side
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Estimation Tool")

                size_input = gr.Number(
                    label="Property Size (sq ft)",
                    placeholder="Enter property size"
                )

                predict_btn = gr.Button(
                    "Predict Rent",
                    variant="primary"
                )

                rent_output = gr.Textbox(
                    label="Predicted Rent",
                    interactive=False
                )

                predict_btn.click(
                    fn=predict_rent,
                    inputs=size_input,
                    outputs=rent_output
                )

            # Right Side
            with gr.Column(scale=1):

                gr.Markdown("### 👨‍💻 About the Developer")
                gr.Markdown("**Manik**")

                gr.Markdown("### 🏫 College")
                gr.Markdown("Panipat Institute of Engineering and Technology")

                gr.Markdown("### 🛠️ Tools Used")
                gr.Markdown("""
- **Python**
- **Gradio**
- **Scikit-Learn**
- **Joblib**
                """)

# Launch App
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
