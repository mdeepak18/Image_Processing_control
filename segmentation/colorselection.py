import cv2
import numpy as np

def empty(a):
    pass

img = cv2.imread("segmented_plum.jpg")
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 600, 400)
# imgHSV = cv2.cvtColor(img,cv2.COLORBGR2HSV)
cv2.createTrackbar("Hue_min","Trackbars",0,179,empty)
cv2.createTrackbar("Hue_max","Trackbars",179,179,empty)
cv2.createTrackbar("Sat_min","Trackbars",0,255,empty)
cv2.createTrackbar("Sat_max","Trackbars",255,255,empty)
cv2.createTrackbar("Val_min","Trackbars",0,255,empty)
cv2.createTrackbar("Val_max","Trackbars",255,255,empty)

while True:
    h_min = cv2.getTrackbarPos("Hue_min","Trackbars")
    h_max = cv2.getTrackbarPos("Hue_max","Trackbars")
    s_min = cv2.getTrackbarPos("Sat_min","Trackbars")
    s_max = cv2.getTrackbarPos("Sat_max","Trackbars")
    v_min = cv2.getTrackbarPos("Val_min","Trackbars")
    v_max = cv2.getTrackbarPos("Val_max","Trackbars")

    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    Mask = cv2.inRange(img, lower, upper)
    imgFin = cv2.bitwise_and(img,img,mask=Mask)
    cv2.imshow("Disp",imgFin)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    #cv2.waitKey(1)
cv2.imwrite("segmented_plum_final.jpg", imgFin)