import numpy as np
import cv2
import glob
import os


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


camera_matrices = []
distortion_matrices = []
#cap = cv2.VideoCapture('rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645')
for file_name in os.listdir('captured_images/'):
#try:
        #while(True):
    #ret, img = cap.read()
    img = cv2.imread('captured_images/'+file_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img',frame)
    print(file_name) 
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
                
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
                #cv2.imshow('img',img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
	            
#except KeyboardInterrupt:
#        pass	

print("Number of images",len(objpoints))

from random import sample 

for i in range(10):
        sampled = sample(list(range(len(objpoints))),100)
        objpoints_sampled = list(np.array(objpoints)[sampled])
        imgpoints_sampled = list(np.array(imgpoints)[sampled])
        print(len(objpoints_sampled), len(imgpoints_sampled))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_sampled, imgpoints_sampled, gray.shape[::-1], None, None)

        print("Camera matrix:")
        print(mtx)

        print("Distortion matrix:")
        print(dist)
        camera_matrices.append(np.array(mtx))
        distortion_matrices.append(np.array(dist))



