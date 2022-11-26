# Leaf-classifier

This is a university project for machine learning course.  Goal of this project is build a classifier able to understand if a leaf is an ivy,medlar or laurel leave leaf.

## Requirements
- python3
- numpy
- pytorch
- torchvision
- torch
- gradio
- torchvision

Install running:
``` pip3 -r requirements.txt```

## Datasets

Dataset's gathering is part of this project. To do that, I recordered some videos of leaves placed on a white sheet , rotating  device on  different axis.  
Inside *src* folder you can find *01_extract-frames.py* which allow to extract frames (from *config.py* file you can select different parameters I.E frames number). 

If you want use video which I recordered make a request [here](https://drive.google.com/drive/folders/1a3shW2Qh0ecZLA0oT_LrJg4iQ6Ns2CY6?usp=share_link). In this folder you can find Videos and Frames already extracted.

## Models

For this has been used three well-known model: *[AlexNet](hhttps://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)*, *[ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)* and *[SqueezeNet](https://arxiv.org/abs/1602.07360)*.


## Project strcuture

This projects has the following structure:
    
    ├── csv                     # Contain all csv generated
    ├── checkpoint              # Contain parameters for each trained models.
    ├── src                     # Source files.
        ├── Extractions         # Contains scripts for frames extraction
        ├── utils               # Some utils method like datasets split , optimizer etc... .
        ├── Training.py         
        ├── Config.py           All configuration useful to play with models and data.
        
    ├── logs                    # logs generated with tensorBoard.
    ...labels*
    └── README.md


## Run models

If you want to run models . go to root folder and run:

```python -m src.gradio-gui``` you will able to open web GUI .You can upload your image and check out if classifier works


 IMMAGINE
 
 

 





