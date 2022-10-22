import gradio as gr

from script.Inference import predict

# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# demo.launch()  

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3)
            
            ).launch()