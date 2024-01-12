import cv2
import numpy as np

# Initialize a list to store clicked coordinates
img_points = []

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
chessboardSize = (3,4)
obj_points = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
obj_points[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
print (obj_points)

# Mouse callback function to capture coordinates
click_count = 0
def mouse_callback(event, x, y, flags, param):
    global click_count
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the pixel color at the current (x, y) coordinates
        # print(f"Mouse Hover - X: {x}, Y: {y}")
        pass
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left mouse button clicked, add the coordinates to the list
        click_count += 1
        print(f"{click_count}. Clicked at (x, y): ({x}, {y})")
        img_points.append((x, y))
        
# Read the image
PATH = 'test_images/f1_test.jpg'
image = cv2.imread(PATH)  # Replace 'your_image.jpg' with the image file path

# Create a window and set the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF
    
    if chessboardSize[0]*chessboardSize[1] == click_count:
        break

    # Press 'q' to exit the loop
    if key == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

# Print the list of clicked coordinates
print("List of clicked coordinates:", img_points)

img_points = np.array([
[310, 1149],
[434, 1148],
[563, 1153],
[347, 1037],
[464, 1038],
[580, 1038],
[384, 939],
[492, 941],
[602, 944],
[411, 856],
[513, 859],
[617, 861]])


img_points = np.array(img_points).astype(np.float32).reshape((-1, 1, 2))
print (obj_points.shape)
print (img_points.shape)
print (image.shape[:-1])
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points], image.shape[:-1], None, None)

print (cameraMatrix)
print (rvecs)
print (tvecs)

### transform image_points to obj_points using cameraMatrix, rvecs and tvecs ###

# Create a 3x3 rotation matrix from the rotation vector
R, _ = cv2.Rodrigues(rvecs[0])

# Combine R and tvec to form the 3x4 extrinsic matrix [R | tvec]
intrinsic_matrix = cameraMatrix
extrinsic_matrix = np.hstack((R, tvecs[0]))
projection_M = intrinsic_matrix @ extrinsic_matrix

ones = np.ones((obj_points.shape[0], 1), dtype=img_points.dtype)
homogeneous_obj_points = np.concatenate([obj_points, ones], axis=1)
print ("homogeneous_obj_points.shape", homogeneous_obj_points.shape)
print (homogeneous_obj_points)
res = projection_M @ homogeneous_obj_points.T
print (res.T)
for coord in res.T:
    print (coord / coord[2]) 
# exit(1)


# ref: https://stackoverflow.com/questions/55734940/how-to-perform-2d-to-3d-reconstruction-considering-camera-calibration
print ('------ recover 3D obj points from 2D img points')

# Convert image point to homogeneous coordinates
img_points = img_points.reshape((-1, 2))
ones = np.ones((img_points.shape[0], 1), dtype=img_points.dtype)
homogeneous_img_point = np.concatenate([img_points, ones], axis=1)
print ("--- homogeneous_img_point --- ")
print (homogeneous_img_point)

# Calculate the final object point
projection_M = np.delete(projection_M, [2], 1)
inv_projection_M = np.linalg.inv(projection_M)
recover_object_points = inv_projection_M @ homogeneous_img_point.T
print("Transformed Object Points (3D World Coordinates):")
# print (recover_object_points.T)

recover_object_points = np.divide(recover_object_points, recover_object_points[2])
recover_object_points[2] = 0
# print (recover_object_points.T)

for img_coord, world_coord in zip(img_points, recover_object_points.T):
    print ("{} -> {}".format(img_coord, world_coord)) 
