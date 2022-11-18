import gradio as gr
import os

from src.Inference import predict
from src.config import GRADIO_LOAD_CHECKPOINT_PATH

def models_items():
    items = os.listdir(GRADIO_LOAD_CHECKPOINT_PATH)
    return items


gr.Interface(fn=predict, 
             inputs=[gr.Image(),gr.Dropdown(models_items())],
             outputs=gr.Label(num_top_classes=3)
            ).launch()