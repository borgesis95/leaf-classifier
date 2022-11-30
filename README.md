# Leaf-classifier

This is a university project for machine learning course.  Goal of this project is build a classifier able to understand if a leaf is an ivy,medlar or laurel leave leaf.

## Requirements

This project requires several libraries like *torchvision,gradio etc..* . You can install them , running following command:

``` pip3 install -r requirements.txt```

## Datasets

Dataset's gathering is part of this project. To do that, I recordered some videos of leaves placed on a white sheet , rotating  device on  different axis.  
Inside *src* folder you can find *01_extract-frames.py* which allow to extract frames (from *config.py* file you can select different parameters like frames number). 

If you want use video which I recordered make a request [here](https://drive.google.com/drive/folders/1a3shW2Qh0ecZLA0oT_LrJg4iQ6Ns2CY6?usp=share_link). In this folder you can find videos and frames already extracted.

## Models

For this has been used three well-known model: *[AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)*, *[ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)* and *[SqueezeNet](https://arxiv.org/abs/1602.07360)*.


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


## Run project

Since Datasets and frames folder are too big to load on Github, I decided to put them on [GDrive](https://drive.google.com/drive/folders/1a3shW2Qh0ecZLA0oT_LrJg4iQ6Ns2CY6?usp=sharing) . If you want start from scratch you have to put *Dataset* folder inside *Root* of this project. You can 
set some parameters on *config*. Now you're ready to run:

```python -m src.extraction.01_frames``` 

If everything work fine, new folder *frames* will be created (unless you decided from config file to change destination folder). 
To create CSV files  run :
``` python -m src.extraction.02_csv ```

After running this scripts, CSV for *train/validation/test set* will be created under csv folder. (Again, you can tuning parameters for creation in config file).

Finally, for launch training phase run:

```python -m src.Main```

## Run Gradio

If you want to run models . go to root folder and run:

```python -m src.gradio-gui``` you will able to open web GUI .You can upload your image and check out if classifier works


 
 
 

 





