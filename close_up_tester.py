# USAGE
# python close_up_tester.py (video url) (folder url where frames are to be stored) (desired frame rate)

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from PIL import Image
import os
import glob
from subprocess import call
import sys

def videoSplitter(videoUrl, folderUrl, frameRate):
    #split video into frames at desired frame rate
    call("ffmpeg -i " + videoUrl + " -r " + frameRate + " -f image2 " + folderUrl +"\image%03d.jpg")

def frameReader(folderUrl):
    frameList=[]
    for imageUrl in glob.glob(os.path.join(sys.argv[1], '*.jpg')):
        frameList.append(imageUrl)
    return frameList

def closeUpDetector(frameList, detector, predictor):
    # set counter for indexing detected images
    count=1
    for (imageCount, frameUrl) in enumerate(frameList):
        image = cv2.imread(frameUrl)
        photoWidth, photoHeight = Image.open(frameUrl).size
  
        image = imutils.resize(image, width=700)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image and store them in an array
        detectedFaces = detector(grayImage, 1)

        # mark all the detected faces and label them as close up or not
        closeUpMarker(detectedFaces, predictor, image, photoWidth, photoHeight, count)
        count+=1

def closeUpMarker(detectedFaces, predictor, image, photoWidth, photoHeight, count ):
    for (faceCount, rect) in enumerate(detectedFaces):
        facialArea=0
        facialScore=0
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(grayImage, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
        (xFace, yFace, faceWidth, faceHeight) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (xFace, yFace), (xFace + faceWidth, yFace + faceHeight), (0, 255, 0), 2)

        # mark the face number
        cv2.putText(image, "Face #{}".format(faceCount + 1), (xFace - 10, yFace - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # determine the close up score to decided whether close up or not            
        facialArea=(faceWidth*faceHeight)
        closeUpScore=(facialArea/(photoWidth*photoHeight))*100

        # mark whether the face is close up or not and save them
        if closeUpScore<1.67:
            cv2.putText(image, "Not a close up", (xFace + 60, yFace - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if faceCount+1==len(detectedFaces):
                cv2.imwrite("./images/close_up/image%03d.jpg" % count, image)
            
        if closeUpScore>=1.67:
            cv2.putText(image, "Close up", (xFace + 60, yFace - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if faceCount+1==len(detectedFaces):
                cv2.imwrite("./images/close_up/image%03d.jpg" % count, image) 

if __name__ == '__main__':
    if(len(sys.argv)<4):
        print("USAGE: python close_up_tester.py (url of folder where frames are to be saved) (url of video to be splitted) (desired frame rate)")
        exit()
    
    # read provided arguments
    folderUrl = sys.argv[1]
    videoUrl = sys.argv[2]
    frameRate = sys.argv[3]

    # call the video splitter utility function
    videoSplitter(videoUrl, folderUrl, frameRate)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # get list of frame images in an array
    frameList=[]
    frameList=frameReader(folderUrl)

    # detect faces in all frame images
    closeUpDetector(frameList, detector, predictor)









        
            

