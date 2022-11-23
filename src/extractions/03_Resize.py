import cv2
from glob import glob 
from os.path import basename
img_paths = glob('.\\frames\\*\\*')
for path in img_paths:
    img = cv2.imread(path)
    new_img = cv2.resize(img,(224,224))
    cv2.imwrite(path,new_img)
    print(basename(path))