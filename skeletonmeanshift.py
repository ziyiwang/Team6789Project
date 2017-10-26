
import numpy as np
import cv2
import time
import math

current_pos = None
tl = None
br = None
green = (0, 255, 0)

def get_rect(im, title='get_rect'):
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)

cap = cv2.VideoCapture('002.avi')

count = 0
ret, frame = cap.read()
while True:
    ret, frame = cap.read()
    if ret == True and count == 10:
        break
    count+=1
#r,h,c,w = 250,90,400,125  # simply hardcoded the values


former_point = 0
a1,a2 = get_rect(frame, title='get_rect')

r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]
c, r, w, h = 154, 70, 12, 22
track_window = (c, r, w, h)

print(c, r, w, h)
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

center_point = []
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
orb = cv2.ORB_create()
while(1):
    ret ,frame = cap.read()
    keypointframe = frame.copy()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        #draw track window
        croplengthint = 50
        croppedimage = keypointframe[y-croplengthint:y + h +croplengthint, x-croplengthint + w // 2:x + w+croplengthint]
        #image cropped run orb detector
        # Initiate STAR detector
        # find the keypoints with ORB
        kp = orb.detect(croppedimage, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(croppedimage, kp)
        # draw only keypoints location,not size and orientation
        #outimage = img
        croppedimage = cv2.drawKeypoints(croppedimage, kp, croppedimage, color=(0, 255, 0), flags=0)
        #plt.imshow(img2), plt.show()
        #cv2.imshow("imagewithkeypoints", img2)

        cv2.imshow("croppedimage",croppedimage)
        center_point.append((x+w//2,y+h//2))
        cv2.circle(frame, (x+w//2,y+h//2), 4, (0, 255, 255), 2, 8)
        #draw rectangle of detecetd frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # original frame


        if former_point!= 0:
            x1, y1 = former_point
            x2, y2 = (x+w//2,y+h//2)
            draw_point = x2 + 6 * (x2 - x1), y2 + 6 * (y2 - y1) # this is the point line ?
            #draw perpendicular line
            #xp1,yp1 = x2-x1, 0
            #xp2,yp2 = x2-x1, y2
            vx = x1 - x2
            vy = y1 - y2
            mag = math.sqrt(vx*vx + vy*vy)
            if (mag != 0):
                print(vx,vy,mag)
                vx = vx //mag
                vy = vy //mag
                temp = vx
                vx = -vy
                vy = temp
                lengthperp = 15
                cx = x1 + vx * lengthperp
                cy = y1 + vy * lengthperp
                dx = x1 + vx *(-lengthperp)
                dy = y1 + vy * (-lengthperp)
                print (cx,cy,dx,dy)
                cv2.line(keypointframe, draw_point, former_point, (255, 0, 0), 3)
                cv2.line(keypointframe, (int(cx), int(cy)), (int(dx), int(dy)), (0, 0, 255), 3)
            #draw_point = xp2 + 14 * (x2 - x1), y2 + 14 * (y2 - y1)  # this is the point line ?
            cv2.line(frame, draw_point, former_point, (255, 0, 0),3)

            cv2.line(frame, (int(cx),int(cy)), (int(dx),int(dy)), (0, 0, 255), 3)

            #cv2.line(frame,(xp1,yp1),(xp2,yp2),(0, 0, 255),3)


        for l in range(len(center_point) - 1):
            x1, y1 = center_point[l + 1]
            x2, y2 = center_point[l]

            cv2.line(frame, center_point[l], center_point[l + 1], green, 2)


        cv2.imshow('img2',frame)
        cv2.imshow('keypoint detection', keypointframe)
        time.sleep(0.2)
        former_point = (x+w//2,y+h//2)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        #else:
           # cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

    cv2.waitKey(0)











cv2.destroyAllWindows()
cap.release()
