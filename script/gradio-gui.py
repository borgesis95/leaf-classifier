import gradio as gr
import os

from script.Inference import predict


def models_items():
    items = os.listdir('checkpoint_13_11')
    return items


gr.Interface(fn=predict, 
             inputs=[gr.inputs.Image(),gr.Dropdown(models_items())],
             outputs=gr.Label(num_top_classes=3)
            ).launch()