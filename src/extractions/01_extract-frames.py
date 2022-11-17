from sys import path
import cv2
import os

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
    videoPath = 'dataset/'
    frames_per_video = 145

    labels_train = open('./labels.train.txt','a')
    labels_test = open('./labels.test.txt','a')

    numbers_of_video = 18

    for curr_folder in (source):

        path = videoPath  + curr_folder
        count_tt = 0
        with os.scandir(path) as it:

            training_percentage = 0.6

            for entry in it:
                folder =""
                if(count_tt/numbers_of_video > training_percentage):
                    print("vai in test")
                    folder = "test/"
                    labels = labels_test
                else :
                    print("vai in train")
                    folder = "train/"
                    labels = labels_train
                print(entry.name,entry.path)
                get_frame(entry.path,'./frames/'+folder,frames_per_video,labels,class_dictionary[curr_folder])
                count_tt = count_tt + 1
            
            print("COUNT FINALE",count)

    labels.close()
    print('All frames are extracted')
    
