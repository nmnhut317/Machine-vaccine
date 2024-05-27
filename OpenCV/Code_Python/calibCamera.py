import numpy as np
import cv2 as cv
import glob


#   FIND CHESSBOARD CORNERS
# chessboardSize = (24, 17) # a number of interesion corners follow width and heigth
# frameSize = (1440, 1080) # pixel of cameras

chessboardSize = (15, 9)
frameSize = (400, 200) # pixel of cameras
print(frameSize)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0: chessboardSize[1]].T.reshape(-1,2)

# sizeof_checkboard = 10
# objp = sizeof_checkboard*objp

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('images/*.png')

images = glob.glob('a/*.jpg')

a_number_of_images = 1

for fname in images:

    # print(fname)
    # imagePath = 'D:\Project\images{}.png'.format(n)

    img = cv.imread(fname)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

        h, w = img.shape[:2]
        # print(img.shape)
        img = cv.resize(img, (0, 0), fx= 0.5, fy= 0.5)
        
        cv.imshow('img', img)

        cv.waitKey(1000)

    a_number_of_images += 1

print('a_number_of_images:', a_number_of_images)

cv.destroyAllWindows()

# calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# print("Camera Calibrate:", ret)
print("\ncameraMatrix", cameraMatrix)
print("\nDistortion Parameters:", dist)
# print("\nRotation Vector:", rvecs)
# print("\nTranstation Vector:", tvecs)

# pickle.dump((cameraMatrix, dist), open('calibration.pkl', "wb"))
# pickle.dump(cameraMatrix, open('cameraMatrix.pkl', "wb"))
# pickle.dump(dist, open('calibration.pkl', "wb"))

k = 1
#               the same feature            #
# THE FIRST METHOD
# Undistortion  
img = cv.imread('D:/Project/a/imaget0.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('D:/Project/a/calibresult{}.jpg'.format(k), dst)
k += 1

#THE SECOND METHOD
# undistort
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('D:/Project/a/calibresult{}.jpg'.format(k), dst)

#               the same feature            #


# check error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )