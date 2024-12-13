import cv2
import numpy as np

# Load camera matrix and distortion coefficients from CSV
camera_matrix = np.genfromtxt('calibration.csv', delimiter=',')
dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

# Define the chessboard pattern size and axis for reprojection
pattern_size = (9, 6) 
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

# Function to draw the 3D axis
def draw(img, originpts, imgpts):
    origin = tuple(originpts[0].ravel().astype(int))
    img = cv2.line(img, origin, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)  # X-axis in Blue
    img = cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)  # Y-axis in Green
    img = cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)  # Z-axis in Red
    return img

# Capture the chessboard pattern and process images
cap = cv2.VideoCapture(2)  # Open the camera (modify index if necessary)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Solve PnP to obtain rotation and translation vectors
        objp = np.zeros((np.prod(pattern_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
        
        # Project 3D points to the image plane
        projpoints, _ = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeffs)
        
        # Draw the 3D axis on the chessboard
        frame = draw(frame, corners2, projpoints)
    
    # Display the processed frame
    cv2.imshow("Pose Estimation", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
