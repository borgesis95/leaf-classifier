from sys import path
import cv2
import os
from src.config import VIDEO_PATH,FRAMES_PER_VIDEO,FRAMES_FOLDER,DISTINCT_TRAINTEST_SET,TRAINING_RATE,DISTINCT_TRAINTEST_SET,LABELS_DISTINCT_LIST,LABELS_ALL,TRAIN_FOLDER,TEST_FOLDER

count = 0

# This method get frames from video and save records <image_path,labels> on a file
def get_frame(path,imagePath,framesNumber,textFile,labelClass):
    global count
    video = cv2.VideoCapture(path)
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frames = totalFrames // framesNumber    

    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    
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

    print("Start Frame extraction:")   

    class_dictionary = {
        "alloro" : 0,
        "edera": 1,
        "nespole": 2
    }
    source = ['alloro','edera','nespole']
    videoPath = VIDEO_PATH 

    frames_per_video = FRAMES_PER_VIDEO

    if(DISTINCT_TRAINTEST_SET == True):

        labels_train = open('./'+ LABELS_DISTINCT_LIST[0],'a')
        labels_test = open('./'+ LABELS_DISTINCT_LIST[1],'a')
    else:
        labels_all = open('./'+ LABELS_ALL,'a')

    numbers_of_video = 18

    for curr_folder in (source):

        path = videoPath  + curr_folder
        count_tt = 0
        with os.scandir(path) as it:


            for entry in it:
                folder =""
                if(DISTINCT_TRAINTEST_SET == True):
                    if(count_tt/numbers_of_video > TRAINING_RATE):
                        print("vai in test")
                        folder = TEST_FOLDER +'/'
                        labels = labels_test
                    else :
                        print("vai in train")
                        folder = TRAIN_FOLDER +'/'
                        labels = labels_train
                else: 
                    labels = labels_all
                
                print(entry.name,entry.path)
                get_frame(entry.path, FRAMES_FOLDER + folder,frames_per_video,labels,class_dictionary[curr_folder])
                count_tt = count_tt + 1
            

    labels.close()
    print('All frames are extracted')
    
