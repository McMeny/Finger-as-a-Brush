import numpy as np
import cv2
import glob

# 10 * 7 = 70; total no. of corners - 70 * 3;
# makes a table with 70 rows, one per corner, and 3 dimensions (x,y,z), all filled with zeros (for now).
chessboard_size = (10, 7) # board has 10 inner corners, 7 down.
objp = np.zeros((np.prod(chessboard_size), 3), np.float32) 

# fills in the x and y columns with grid positions. Grid position - each corners address on the board such as its row and column.
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)


# arrays to store object points and image points
obj_points = [] # corner point position on the actual physical board
img_points = [] # corner point position in the camera picture, measured in pixels

# overall, until this point:
# builds a list of where every inner corner is on the real, flat chessboard. 
# Calibration later compares this "real" list with where the corners appear in the camera image.

#load all images for calibration

images = glob.glob(r'C:\Users\minua\OneDrive\Pictures\Camera Roll\opencv_calibration\*.jpg')
print(f"No. of images found: {len(images)}")
detected = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to open image: {fname}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ret: True or False; whether OpenCV foundd the whole chessboard in photo.
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        obj_points.append(objp)
        img_points.append(corners)
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        print(f"Chessboard corners DETECTED!")
        detected += 1
    else:
        print(f"Chessboard corners not detected in: {fname}")
        
print(f"No. of Chessboard corners detected: {detected}")

cv2.destroyAllWindows()

# Now, perform camera calibration if at least one valid image was found.
# None, None = "I have no starting guess for the camera matrix or distortion. Work them out from scratch."
# rvecs, tvecs = Where the chessboard was (rotation and position) in each photo. 
if len(obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    
# Outputs of calibration that are useful are camera_matrix and dist_coeffs. Save to a file:
np.savez("calibration.npz", camera_matrix = camera_matrix, dist_coeffs = dist_coeffs)