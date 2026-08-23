"""Gradio web demo: upload a blood cell image, get a cancerous / non-cancerous prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gradio as gr

from config import CHECKPOINT_PATH
from predict import load_model, predict_image

DISCLAIMER = (
    "**Research/educational demo only — not a medical device.** "
    "This model is trained on a public research dataset (C-NMC 2019 leukemia "
    "blood-smear images) and must not be used for real diagnosis. Always consult "
    "a qualified pathologist."
)


def classify(image):
    if image is None:
        return {}, "Upload an image first."
    if not CHECKPOINT_PATH.exists():
        return {}, f"No trained model found at {CHECKPOINT_PATH}. Run `python src/train.py` first."

    predicted_class, probs = predict_image(image)
    label = "Cancerous (malignant)" if predicted_class == "cancerous" else "Non-cancerous (normal)"
    return probs, label


def build_interface():
    if CHECKPOINT_PATH.exists():
        load_model()

    with gr.Blocks(title="Blood Cell Cancer Detection") as demo:
        gr.Markdown("# Blood Cell Cancer Detection")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Blood cell microscopy image")
                submit_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                label_output = gr.Textbox(label="Prediction")
                probs_output = gr.Label(label="Class probabilities", num_top_classes=2)

        submit_btn.click(fn=classify, inputs=image_input, outputs=[probs_output, label_output])
        image_input.change(fn=classify, inputs=image_input, outputs=[probs_output, label_output])

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
