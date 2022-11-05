import gradio as gr
from script.Main import train
def train(lr,model,epochs,da,fe,external):

    print("START")
    accuracy = train(model,True,external=external,epochs=int(epochs),lr=lr,feature_extr=fe,data_augmentation=da)
    return "pippo"

demo = gr.Interface(
    fn=train,
    inputs=[
        gr.Dropdown([0.2,0.02,0.002],label="Choose LR"),
        gr.Dropdown(["AlexNet","ResNet","SqueezeNet"], label="Choose model"),
        gr.Number(label="Choose epochs number"),
        gr.Checkbox(label="Enable data augmentation"),
        gr.Checkbox(label="Enable Feature extraction"),
        gr.Checkbox(label="External")

    ],
    outputs="text",
    
)

demo.queue(concurrency_count=5, max_size=20).launch()