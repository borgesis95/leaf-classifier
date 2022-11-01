import gradio as gr
import os

from script.Inference import predict


def models_dropdown():
    items = os.listdir('checkpoint')
    return items


gr.Interface(fn=predict, 
             inputs=[gr.inputs.Image(),gr.Dropdown(models_dropdown())],
             outputs=gr.Label(num_top_classes=3)
            ).launch()