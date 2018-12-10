# USAGE
# python close_up_tester.py (video url) (folder url where frames are to be stored) 

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

#split video into frames at desired frame rate
call("ffmpeg -i " + sys.argv[2] + " -r 3 -f image2 " + sys.argv[1] +"\image%03d.jpg")

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
# ap.add_argument("-i", "--folder", required=True, help="path to input frames folder")
# ap.add_argument("-v", "--video", required=True, help="video address")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#required parameters
area=0
no_count=0
yes_count=0
temp_list=[]
is_close_up_list=[]
not_close_up_list=[]
score=0
x=0


for image in glob.glob(os.path.join(sys.argv[1], '*.jpg')):
    temp_list.append(image)

for image_url in temp_list:
    image = cv2.imread(image_url)
    im = Image.open(image_url)
    width, height = im.size
    # print(width, height)
    image = imutils.resize(image, width=700)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        area=0
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
            
        area=area+(w*h)
        #print(area, w*h, width*height, (area/(width*height))*100)
        score=(area/(width*height))*100
        if score<1.67:
            not_close_up_list.append(image_url)
            cv2.putText(image, "Not a close up", (x + 60, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if i+1==len(rects):
                cv2.imwrite("./images/close_up/image%d.jpg" % x, image)
            x+=1
        if score>=1.67:
            is_close_up_list.append(image_url)
            cv2.putText(image, "Close up", (x + 60, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if i+1==len(rects):
                cv2.imwrite("./images/close_up/image%d.jpg" % x, image)
            x+=1    
    # area=0
        # show the output image with the face detections + facial landmarks
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)


        #im = Image.open(args["image"])
        #width, height = im.size
        #print(width, height)


        
            

