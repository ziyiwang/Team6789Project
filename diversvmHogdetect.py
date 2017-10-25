# USAGE
# python detectmultiscale.py --image images/person_010.bmp

# import the necessary packages
from __future__ import print_function
import argparse
import datetime
import imutils
import cv2
import numpy as np


# Function definition is here
def changeme( mylist ):
  # "This changes a passed list into this function"
    mylist.append([1,2,3,4])
    print("Values inside the function: ", mylist)
    return

#create svms for training

def get_svm_detector(svm, hog_detector) :

    sv =  np.zeros((256, 256, 1), dtype = "uint8")
    sv = svm.getSupportVector()
    sv_total = sv.rows
    alpha = np.zeros((256, 256, 1), dtype = "uint8")
    svidx = np.zeros((256, 256, 1), dtype = "uint8")
    rho = svm.getDecisionFunction(0,alpha, svidx)




SVM = cv2.ml.SVM_create()
SVM.setKernel(cv2.ml.SVM_LINEAR)
SVM.setP(0.2)
SVM.setType(cv2.ml.SVM_EPS_SVR)
SVM.setC(1.0)

#training
#SVM.train_auto(samples, cv2.ml.ROW_SAMPLE, responses)

#predict
#output = SVM.predict(samples)[1].ravel()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the input image")
ap.add_argument("-w", "--win-stride", type=str, default="(4, 4)",
	help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
	help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.5,
	help="image pyramid scale")
ap.add_argument("-m", "--mean-shift", type=int, default=-1,
	help="whether or not mean shift grouping should be used")
args = vars(ap.parse_args())

# evaluate the command line arguments (using the eval function like
# this is not good form, but let's tolerate it for the example)
winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hoge = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.load("C:\devFolder\opencvassProj\SURF_test\OpenCV_test\\x64\Debug\HOGdiver96x160.yml")
# load the image and resize it
hoge.load("C:\devFolder\opencvassProj\SURF_test\OpenCV_test\\x64\Debug\HOGdiverwaterentry.yml");

#read from the video file
#cap = cv2.VideoCapture('C:\devFolder\opencvassProj\SURF_test\OpenCV_test\OpenCV_test\outputVideo.mp4')
#cap = cv2.VideoCapture('C:\devFolder\group6789\dataset\dataset\\1\\002.avi')
cap = cv2.VideoCapture(args["video"])

#image = cv2.imread(args["image"])
start = datetime.datetime.now()
#(rects, weights) = hog.detectMultiScale(image, winStride=winStride,padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
#print("[INFO] detection took: {}s".format(
#    (datetime.datetime.now() - start).total_seconds()))
#0
# draw the original bounding boxes
#for (x, y, w, h) in rects:
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
#cv2.imshow("Detections", image)
#cv2.waitKey(0)
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
beginning = length // 4
ending = length - (length // 2)
frameTrack = 0

while(cap.isOpened()):
    frameTrack=frameTrack+1
    ret, frame = cap.read()
    #1st if statement
    if (frameTrack <= beginning):
       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = frame
      #  image = cv2.imread(args["image"])
        start = datetime.datetime.now()
        height, width, channels = image.shape
      #  crop_img = image[30:(height-30), 30:(width-30)]
      #  bkp_img = image
      #  image = crop_img
       # cv2.imshow("cropped image", image)
        (rects, weights) = hog.detectMultiScale(image, hitThreshold=0.4,winStride=winStride,
                                                padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
        print("[INFO] detection took: {}s".format(
            (datetime.datetime.now() - start).total_seconds()))

        # draw the original bounding boxes
        for (x, y, w, h) in rects: #detections for jump start
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # original frame
            #if whole frame needed for K clustering, use above line image
            cv2.putText(image, "Diver is jumping", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            croppedimage = image[y:y+h-35,x+w//2:x+w]
            cv2.imshow("cropped",croppedimage)


        # show the output image
        cv2.imshow("Detections", image)
        cv2.waitKey(0)
    elif (frameTrack >= ending and frameTrack <= length):

       # cv2.imshow('frame',gray)
       # if cv2.waitKey(1) & 0xFF == ord('q'):
       #     break
       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       image = frame
       #  image = cv2.imread(args["image"])
       start = datetime.datetime.now()
       height, width, channels = image.shape
       #  crop_img = image[30:(height-30), 30:(width-30)]
       #  bkp_img = image
       #  image = crop_img
       #cv2.imshow("cropped image", image)
       (rects, weights) = hoge.detectMultiScale(image, hitThreshold=0.08, winStride=winStride,
                                               padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
       print("[INFO] detection took: {}s".format(
           (datetime.datetime.now() - start).total_seconds()))

       # draw the original bounding boxes
       for (x, y, w, h) in rects:
           cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
           cv2.putText(image, "Water entry", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
         #  croppedimage = image[y:y + h - 35, x + w // 2:x + w]
          # cv2.imshow("cropped", croppedimage)

       # show the output image
       cv2.imshow("Detections", image)
       cv2.waitKey(0)
    #in the middle here

    else:

        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        # end in the middle

cap.release()
#cv2.destroyAllWindows()






#image = cv2.imread(args["image"])
#image = imutils.resize(image, width=min(400, image.shape[1]))




# detect people in the image
