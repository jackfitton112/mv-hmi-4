import cv2
import numpy as np

# Load camera calibration matrix and distortion coefficients
camera_matrix = np.genfromtxt('calibration.csv', delimiter=',')
dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

# Initialize variables
captured_images = []
image_count = 0

# Function to save the captured images
def save_image(image, count):
    filename = f"image_{count}.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved {filename}")

# Capture images from the camera
cap = cv2.VideoCapture(2)
cv2.namedWindow("Capture")

print("Press 'c' to capture an image. Press 'ESC' to exit.")

while image_count < 2:  # Capture exactly two images
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Undistort the captured image
    h, w = frame.shape[:2]
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)
    
    # Display the current frame
    cv2.imshow("Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Press 'c' to capture an image
        captured_images.append(undistorted)
        image_count += 1
        save_image(undistorted, image_count)
    elif key == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyWindow("Capture")

if image_count < 2:
    print("Not enough images captured. Exiting.")
    exit()

print("Two images have been captured and saved.")

# Stitch the captured images
print("Stitching images...")

# Create a Stitcher object
stitcher = cv2.Stitcher_create() if int(cv2.__version__.split('.')[0]) >= 4 else cv2.createStitcher()

# Perform stitching
status, stitched_image = stitcher.stitch(captured_images)

if status == cv2.Stitcher_OK:
    print("Stitching successful.")
    cv2.imshow("Stitched Image", stitched_image)
    cv2.imwrite("stitched_result.jpg", stitched_image)
    print("Stitched image saved as 'stitched_result.jpg'.")
else:
    print(f"Stitching failed with status code {status}.")

cv2.waitKey(0)
cv2.destroyAllWindows()
