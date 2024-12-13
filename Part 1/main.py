# Camera Calibration Script using open CV and Detect Chessboard Corners

import cv2
import numpy as np
import time
import os
import csv

IMAGES_FOLDER = "images"


#if there are more than 10 images in the folder try to calibrate the camera, if not as if the
#user wants to take more photos
#num_of_images = len(os.listdir(IMAGES_FOLDER))

#if num_of_images < 10:
#    print("Please take more photos of the chessboard")



def take_photo():

    # Define the number of inside corners in x and y direction
    # Chessboard has 7 rows and 10 columns
    nx = 9
    ny = 6

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)

    # Fill the object points with the coordinates of the chessboard corners
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    #set up camera
    cap = cv2.VideoCapture(4)

    # Read in the images
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #show video stream
        cv2.imshow("Stream", gray)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img', img)
            #save
            file = f"images/Chess-{int(time.time())}.jpg"
            cv2.imwrite(file, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calibrate_camera():
    # Define the number of inside corners in x and y direction
    # Chessboard has 7 rows and 10 columns
    nx = 9
    ny = 6

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)

    # Fill the object points with the coordinates of the chessboard corners
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    #set up camera
    cap = cv2.VideoCapture(4)

    #print number of images
    print(f"Number of images: {len(os.listdir(IMAGES_FOLDER))}")
    startTime = time.time()

    # Read in the images
    for file in os.listdir(IMAGES_FOLDER):

        if ".jpg" not in file:
            continue

        img = cv2.imread(f"{IMAGES_FOLDER}/{file}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img', img)
            #save
            file = f"images/Chess-{int(time.time())}.jpg"
            cv2.imwrite(file, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #print the camera matrix and distortion coefficients
    print("Camera Matrix: ", mtx)
    print("Distortion Coefficients: ", dist)
    print("Rotation Vectors: ", rvecs)
    print("Translation Vectors: ", tvecs)

    #save all the calibration data to a file including vectors
    np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    #save data as csv
    with open("images/calibration.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(mtx)


    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    print("Calibration Completed in: ", time.time() - startTime)


    return mtx, dist

calibrate_camera()

#take_photo()