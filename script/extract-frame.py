from sys import path
import cv2
from zipfile import ZipFile
import os
from os.path import basename

count = 0
def get_frame(path,imagePath,framesNumber,textFile,labelClass):
    global count
    video = cv2.VideoCapture(path)
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = totalFrames // framesNumber

    for i in range(framesNumber):
        video.set(1,i*frames)
        ret, frame = video.read()

        if ret == False:
            break;

        textFile.write('leaf_%d.jpg, %d\n' %(count,labelClass))
        cv2.imwrite(imagePath + 'leaf_' + str(count) + '.jpg',frame)
        count = count + 1
    video.release()
    return True



if __name__ =="__main__":

    class_dictionary = {
        "alloro" : 1,
        "mandarino": 2,
        "ulivo": 3
    }
    source = ['alloro','mandarino','ulivo']
    videoPath = 'dataset/'

    frames_per_video = 1
    labels = open('./labels.txt','a')

    for curr_folder in (source):
        print("curr_folder",curr_folder)
        path = videoPath  + curr_folder
        with os.scandir(path) as it:
            for entry in it:
                print(entry.name,entry.path)
                get_frame(entry.path,'./frame/',frames_per_video,labels,class_dictionary[curr_folder])

    labels.close()
    print('All frames are extracted')
    
