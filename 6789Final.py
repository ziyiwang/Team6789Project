
# USAGE
# python detectmultiscale.py --image images/person_010.bmp

# import the necessary packages
from __future__ import print_function
import argparse
import datetime
import imutils
import cv2
import time
import numpy as np
import math

# Function definition is here
def changeme(mylist):
    # "This changes a passed list into this function"
    mylist.append([1, 2, 3, 4])
    print("Values inside the function: ", mylist)
    return


# create svms for training

def get_svm_detector(svm, hog_detector):
    sv = np.zeros((256, 256, 1), dtype="uint8")
    sv = svm.getSupportVector()
    sv_total = sv.rows
    alpha = np.zeros((256, 256, 1), dtype="uint8")
    svidx = np.zeros((256, 256, 1), dtype="uint8")
    rho = svm.getDecisionFunction(0, alpha, svidx)


# --------------------------------------------Get meanshift window--------------------------------------------- #

def get_window(image):
    kmean_im = kmean_clustering(image)
    crop_im = crop_kmean(kmean_im)
    thresh_im, max_line_row = Thresholding(crop_im)
    window_im, top_coor, bot_coor = draw_window(thresh_im, max_line_row)
    cv2.imshow('window_im', window_im)
    return window_im, top_coor, bot_coor


def kmean_clustering(image):
    Z = image.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2


def crop_kmean(image):
    sz = image.shape
    height = sz[0]
    width = sz[1]

    image[0: (int)(height / 3), :] = 0;
    image[(int)(height - height / 4) - 40: height, :] = 0;
    image[:, 0: (int)(width / 3) + 47] = 0;
    image[:, (int)(width - width / 3) - 10: width] = 0;
    return image


def Thresholding(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalised = cdf * hist.max() / cdf.max()

    height, width, depth = image.shape
    # image[0:height, 0:width//4, 0:depth] = 0
    max_line = 0
    max_line_row = 0
    for r in range(0, height):
        line = 0
        for c in range(0, width):
            if (image[r, c, 0] < 100 or image[r, c, 1] < 110 or image[r, c, 2] < 100):
                # print(image[r,c,:])
                image[r, c, :] = 0
            else:
                image[r, c, :] = 255
                line += 1
            if (c == (width - 1)):
                if (line > max_line):
                    max_line = line
                    max_line_row = r
                    # print(max_line)

    return image, max_line_row


def draw_window(image, max_line_row):
    start_width = 0
    end_width = 0
    first_time = 1
    for c in range(0, width):
        if (image[max_line_row, c, 0] == 255):
            if (first_time == 1):
                start_width = c
                first_time = 0
        if (image[max_line_row, c, 0] == 0):
            if (first_time == 0):
                end_width = c
                first_time = 1
                break

    center_width = start_width + (int)((end_width - start_width) / 2)
    # topleft_col = center_width - 3
    # topleft_row = max_line_row - 10
    # botright_col = center_width + 3
    # botright_row = max_line_row + 10

    topleft_col = center_width - 6
    topleft_row = max_line_row - 5
    botright_col = center_width + 6
    botright_row = max_line_row + 20

    topright_col = center_width + 6
    topright_row = max_line_row - 5
    botleft_col = center_width - 6
    botleft_row = max_line_row + 20

    cv2.rectangle(image, (topleft_col, topleft_row), (botright_col, botright_row), (0, 255, 0), 1)
    cv2.line(image, (start_width, max_line_row), (end_width, max_line_row), (255, 0, 0), 1)
    cv2.circle(image, (center_width, max_line_row), 2, (0, 0, 255), -1)

    top_coor = [topleft_col, topleft_row, topright_col, topright_row]

    bot_coor = [botleft_col, botleft_row, botright_col, botright_row]

    return image, top_coor, bot_coor


# --------------------------------------------------------------------------------------------------------------------------- #


SVM = cv2.ml.SVM_create()
SVM.setKernel(cv2.ml.SVM_LINEAR)
SVM.setP(0.2)
SVM.setType(cv2.ml.SVM_EPS_SVR)
SVM.setC(1.0)

# training
# SVM.train_auto(samples, cv2.ml.ROW_SAMPLE, responses)

# predict
# output = SVM.predict(samples)[1].ravel()

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
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.load("HOGdiver96x160.yml")
# load the image and resize it
hoge.load("HOGdiverwaterentry.yml");

# read from the video file
# cap = cv2.VideoCapture('C:\devFolder\opencvassProj\SURF_test\OpenCV_test\OpenCV_test\outputVideo.mp4')
# cap = cv2.VideoCapture('C:\devFolder\group6789\dataset\dataset\\1\\002.avi')
cap = cv2.VideoCapture(args["video"])

# image = cv2.imread(args["image"])
start = datetime.datetime.now()
# (rects, weights) = hog.detectMultiScale(image, winStride=winStride,padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
# print("[INFO] detection took: {}s".format(
#    (datetime.datetime.now() - start).total_seconds()))
# 0
# draw the original bounding boxes
# for (x, y, w, h) in rects:
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
# cv2.imshow("Detections", image)
# cv2.waitKey(0)
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
beginning = length // 4
ending = length - (length // 2)
frameTrack = 0
water_entry = 0
entry_first = 1
start_meanshift = 0
diving_start = 0
diving_end = 0
orb = cv2.ORB_create()

while (cap.isOpened()):
    frameTrack = frameTrack + 1
    ret, frame = cap.read()
    image = frame
    keypointframe = frame.copy()
    # 1st if statement
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
        (rects, weights) = hog.detectMultiScale(image, hitThreshold=0.4, winStride=winStride,
                                                padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
        print("[INFO] detection took: {}s".format(
            (datetime.datetime.now() - start).total_seconds()))

        # draw the original bounding boxes
        for (x, y, w, h) in rects:  # detections for jump start
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # original frame
            start_meanshift = 1
            # if(entry_first == 1):
            #   window_im, top_coor, bot_coor = get_window(image)
            #   cv2.imshow('window_im', window_im)
            #   entry_first = 0
            # if whole frame needed for K clustering, use above line image
            cv2.putText(image, "Diver is jumping", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            croppedimage = image[y:y + h - 35, x + w // 2:x + w]
            cv2.imshow("cropped", croppedimage)

        # show the output image

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
        # cv2.imshow("cropped image", image)
        (rects, weights) = hoge.detectMultiScale(image, hitThreshold=0.08, winStride=winStride,
                                                 padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
        print("[INFO] detection took: {}s".format(
            (datetime.datetime.now() - start).total_seconds()))

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            water_entry = 1
            cv2.putText(image, "Water entry", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            #  croppedimage = image[y:y + h - 35, x + w // 2:x + w]
            # cv2.imshow("cropped", croppedimage)

        # show the output image

    # in the middle here

    if start_meanshift == 1:
        if entry_first == 1:
            window_im, top_coor, bot_coor = get_window(image)
            green = (0, 255, 0)
            former_point = 0

            now = 0.0
            c, r, w, h= top_coor[0], top_coor[1],12,25

            track_window = (c, r, w, h)
            roi = frame[r:r + h, c:c + w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            center_point = []
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            cv2.imshow('window_im', window_im)
            entry_first = 0
        elif water_entry == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw it on image
            x, y, w, h = track_window
            # draw track window
            croplengthint = 50
            croppedimage = keypointframe[y - croplengthint:y + h + croplengthint,
                           x - croplengthint + w // 2:x + w + croplengthint]

            # image cropped run orb detector
            # Initiate STAR detector
            # find the keypoints with ORB
            kp = orb.detect(croppedimage, None)
            # compute the descriptors with ORB
            kp, des = orb.compute(croppedimage, kp)
            # draw only keypoints location,not size and orientation
            # outimage = img
            croppedimage = cv2.drawKeypoints(croppedimage, kp, croppedimage, color=(0, 255, 0), flags=0)


            cv2.putText(image, "diving", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            center_point.append((x + w // 2, y + h // 2))
            cv2.circle(frame, (x + w // 2, y + h // 2), 4, (0, 255, 255), 2, 8)
            if former_point != 0:
                x1, y1 = former_point
                x2, y2 = (x + w // 2, y + h // 2)
                draw_point = x2 + 4 * (x2 - x1), y2 + 4 * (y2 - y1)
                # draw perpendicular line
                # xp1,yp1 = x2-x1, 0
                # xp2,yp2 = x2-x1, y2
                vx = x1 - x2
                vy = y1 - y2
                mag = math.sqrt(vx * vx + vy * vy)
                if (mag != 0):
                    print(vx, vy, mag)
                    vx = vx // mag
                    vy = vy // mag
                    temp = vx
                    vx = -vy
                    vy = temp
                    lengthperp = 15
                    cx = x1 + vx * lengthperp
                    cy = y1 + vy * lengthperp
                    dx = x1 + vx * (-lengthperp)
                    dy = y1 + vy * (-lengthperp)
                    print(cx, cy, dx, dy)
                    cv2.line(keypointframe, draw_point, former_point, (255, 0, 0), 3)
                    cv2.line(keypointframe, (int(cx), int(cy)), (int(dx), int(dy)), (0, 0, 255), 3)
                cv2.line(frame, draw_point, former_point, (255, 0, 0), 3)

            for l in range(len(center_point) - 1):
                x1, y1 = center_point[l + 1]
                x2, y2 = center_point[l]

                cv2.line(frame, center_point[l], center_point[l + 1], green, 2)

            image = frame
            now += 1.0/30.0
            cv2.putText(image, "diving in " + str(now)[:3] + "s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            former_point = (x + w // 2, y + h // 2)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

            # else:

            #     cv2.imshow("Detections", frame)
            #     cv2.waitKey(0)
            #     # end in the middle
    cv2.imshow("Detections", image)
    cv2.imshow('keypoint detection', keypointframe)
    cv2.waitKey(0)
cap.release()
# cv2.destroyAllWindows()






# image = cv2.imread(args["image"])
# image = imutils.resize(image, width=min(400, image.shape[1]))




# detect people in the image
