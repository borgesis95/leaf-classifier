from sys import path
import cv2
import os

count = 0
def get_frame(path,imagePath,framesNumber,textFile,labelClass):
    global count
    video = cv2.VideoCapture(path)
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = totalFrames // framesNumber    
    print("frames",frames)

    for i in range(framesNumber):
        video.set(1,i*frames)
        ret, frame = video.read()

        if ret == False:
            break;

        textFile.write('leaf_%d_%d.jpg, %d\n' %(labelClass,count,labelClass))
        cv2.imwrite(imagePath + 'leaf_' + str(labelClass) + '_' + str(count) +'.jpg',frame)
        count = count + 1
    video.release()
    return True

if __name__ =="__main__":

    print("Start Frame extraction...")
    class_dictionary = {
        "alloro-2" : 0,
        "edera": 1,
        "nespole": 2
    }
    source = ['alloro-2','edera','nespole']
    videoPath = 'dataset/'

    frames_per_video = 300
    print("Numero di frame per video - ",frames_per_video)
    labels = open('./labels.txt','a')

    for curr_folder in (source):
        path = videoPath  + curr_folder
        with os.scandir(path) as it:
            for entry in it:
                print(entry.name,entry.path)
                get_frame(entry.path,'./frames/',frames_per_video,labels,class_dictionary[curr_folder])

    labels.close()
    print('All frames are extracted')
    
