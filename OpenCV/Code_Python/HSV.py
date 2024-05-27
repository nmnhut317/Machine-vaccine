from pypylon import pylon
import cv2 as cv
import numpy as np
import glob
import pandas as pd
import time
import os
from datetime import datetime
# 82, 105, 51
# 91, 116, 57
# 81, 108, 50
# 88, 113, 59
# 96, 123, 63

# 65, 103, 87


def findContour(threshold):
    gray = cv.cvtColor(threshold, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    contour = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    if len(contour)>0:
        contour = max(contour, key= cv.contourArea) 
        rect = cv.minAreaRect(contour)

    return rect, threshold

def nothing(x):
        pass

cv.namedWindow("namespace")
cv.createTrackbar("lh", "namespace", 0, 255, nothing)
cv.createTrackbar("ls", "namespace", 0, 255, nothing)
cv.createTrackbar("lv", "namespace", 0, 255, nothing)
cv.createTrackbar("hh", "namespace", 255, 255, nothing)
cv.createTrackbar("hs", "namespace", 255, 255, nothing)
cv.createTrackbar("hv", "namespace", 255, 255, nothing)

# duong dan file
path = "D:\\Project\\e\\calibresult11.jpg"

while True:

    img = cv.imread(path)

    img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # xac dinh gia tri hsv thong qua cac thanh truot
    lh = cv.getTrackbarPos("lh", "namespace")
    ls = cv.getTrackbarPos("ls", "namespace")
    lv = cv.getTrackbarPos("lv", "namespace")  
    hh = cv.getTrackbarPos("hh", "namespace")
    hs = cv.getTrackbarPos("hs", "namespace")
    hv = cv.getTrackbarPos("hv", "namespace")

    low_red = np.array([lh, ls, lv])
    high_red = np.array([hh, hs, hv])
    mask = cv.inRange(hsv, low_red, high_red)
    mask = ~mask
    result = cv.bitwise_and(img, img, mask= mask)

    # kiem tra lai anh sau khi tach nen
    # low_red = np.array([65, 116, 95])
    # high_red = np.array([179, 255, 255])

    # mask = cv.inRange(hsv, low_red, high_red)
    # result = cv.bitwise_and(img, img, mask= mask)
    # threshold = ~mask

    # kernel = np.ones((7,7),np.uint8)
    # threshold1 = cv.medianBlur(threshold, 5)
    # threshold2 = cv.morphologyEx(threshold1, cv.MORPH_CLOSE, kernel)
    # contour = cv.findContours(threshold1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    
    # contour = max(contour, key= cv.contourArea)

    # newImg= np.zeros(img.shape, dtype= np.uint8)
    # cv.drawContours(newImg, [contour], 0, (255, 255, 255), -1)

    # newImg = cv.split(newImg)[0]
    # img = cv.bitwise_and(img, img, mask= newImg)


    # cv.imshow("mask", mask)
    cv.imshow("img", img)
    cv.imshow("result", result)
    # cv.imshow("toggle mask", newImg)
    # cv.imshow("filter", threshold)


    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
