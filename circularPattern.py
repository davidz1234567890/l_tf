import numpy as np
import cv2
import glob
import pickle

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################
# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1
blobParams.maxCircularity = 1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87
blobParams.maxConvexity = 1

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01
blobParams.maxInertiaRatio = 1

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)
###################################################################################################


###################################################################################################
# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.
# objp = np.zeros((44, 3), np.float32)
# objp[0]  = (0  , 0  , 0)
# objp[1]  = (0  , 72 , 0)
# objp[2]  = (0  , 144, 0)
# objp[3]  = (0  , 216, 0)
# objp[4]  = (36 , 36 , 0)
# objp[5]  = (36 , 108, 0)
# objp[6]  = (36 , 180, 0)
# objp[7]  = (36 , 252, 0)
# objp[8]  = (72 , 0  , 0)
# objp[9]  = (72 , 72 , 0)
# objp[10] = (72 , 144, 0)
# objp[11] = (72 , 216, 0)
# objp[12] = (108, 36,  0)
# objp[13] = (108, 108, 0)
# objp[14] = (108, 180, 0)
# objp[15] = (108, 252, 0)
# objp[16] = (144, 0  , 0)
# objp[17] = (144, 72 , 0)
# objp[18] = (144, 144, 0)
# objp[19] = (144, 216, 0)
# objp[20] = (180, 36 , 0)
# objp[21] = (180, 108, 0)
# objp[22] = (180, 180, 0)
# objp[23] = (180, 252, 0)
# objp[24] = (216, 0  , 0)
# objp[25] = (216, 72 , 0)
# objp[26] = (216, 144, 0)
# objp[27] = (216, 216, 0)
# objp[28] = (252, 36 , 0)
# objp[29] = (252, 108, 0)
# objp[30] = (252, 180, 0)
# objp[31] = (252, 252, 0)
# objp[32] = (288, 0  , 0)
# objp[33] = (288, 72 , 0)
# objp[34] = (288, 144, 0)
# objp[35] = (288, 216, 0)
# objp[36] = (324, 36 , 0)
# objp[37] = (324, 108, 0)
# objp[38] = (324, 180, 0)
# objp[39] = (324, 252, 0)
# objp[40] = (360, 0  , 0)
# objp[41] = (360, 72 , 0)
# objp[42] = (360, 144, 0)
# objp[43] = (360, 216, 0)

# objp = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [
#                 0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0], [3.5, 0.5, 0]])
# for y in range(2, 11):
#     for x in range(4):
#         objp = np.append(
#             objp, [np.array([objp[4*(y-2)+x][0], objp[4*(y-2)+x][1]+1, 0])], axis=0)


# objp = np.zeros((44, 3), np.float32)
# objp[0] = (0, 0, 0)
# objp[1] = (0, 5, 0)
# objp[2] = (0, 10, 0)
# objp[3] = (0, 15, 0)
# objp[4] = (2.5, 2.5, 0)
# objp[5] = (2.5, 7.5, 0)
# objp[6] = (2.5, 12.5, 0)
# objp[7] = (2.5, 17.5, 0)
# objp[8] = (5, 0, 0)
# objp[9] = (5, 5, 0)
# objp[10] = (5, 10, 0)
# objp[11] = (5, 15, 0)
# objp[12] = (7.5, 2.5,  0)
# objp[13] = (7.5, 7.5, 0)
# objp[14] = (7.5, 12.5, 0)
# objp[15] = (7.5, 17.5, 0)
# objp[16] = (10, 0, 0)
# objp[17] = (10, 5, 0)
# objp[18] = (10, 10, 0)
# objp[19] = (10, 15, 0)
# objp[20] = (12.5, 2.5, 0)
# objp[21] = (12.5, 7.5, 0)
# objp[22] = (12.5, 12.5, 0)
# objp[23] = (12.5, 17.5, 0)
# objp[24] = (15, 0, 0)
# objp[25] = (15, 5, 0)
# objp[26] = (15, 10, 0)
# objp[27] = (15, 15, 0)
# objp[28] = (17.5, 2.5, 0)
# objp[29] = (17.5, 7.5, 0)
# objp[30] = (17.5, 12.5, 0)
# objp[31] = (17.5, 17.5, 0)
# objp[32] = (20, 0, 0)
# objp[33] = (20, 5, 0)
# objp[34] = (20, 10, 0)
# objp[35] = (20, 15, 0)
# objp[36] = (22.5, 2.5, 0)
# objp[37] = (22.5, 7.5, 0)
# objp[38] = (22.5, 12.5, 0)
# objp[39] = (22.5, 17.5, 0)
# objp[40] = (25, 0, 0)
# objp[41] = (25, 5, 0)
# objp[42] = (25, 10, 0)
# objp[43] = (25, 15, 0)

objp = np.zeros((44, 3), np.float32)
objp[0] = (0, 0, 0)
objp[1] = (4, 0, 0)
objp[2] = (8, 0, 0)
objp[3] = (12, 0, 0)
objp[4] = (16, 0, 0)
objp[5] = (20, 0, 0)

objp[6] = (2, 2, 0)
objp[7] = (6, 2, 0)
objp[8] = (10, 2, 0)
objp[9] = (14, 2, 0)
objp[10] = (18, 2, 0)

objp[11] = (0, 4, 0)
objp[12] = (4, 4,  0)
objp[13] = (8, 4, 0)
objp[14] = (12, 4, 0)
objp[15] = (16, 4, 0)
objp[16] = (20, 4, 0)


objp[17] = (2, 6, 0)
objp[18] = (6, 6, 0)
objp[19] = (10, 6, 0)
objp[20] = (14, 6, 0)
objp[21] = (18, 6, 0)


objp[22] = (0, 8, 0)
objp[23] = (4, 8, 0)
objp[24] = (8, 8, 0)
objp[25] = (12, 8, 0)
objp[26] = (16, 8, 0)
objp[27] = (20, 8, 0)


objp[28] = (2, 10, 0)
objp[29] = (6, 10, 0)
objp[30] = (10, 10, 0)
objp[31] = (14, 10, 0)
objp[32] = (18, 10, 0)


objp[33] = (0, 12, 0)
objp[34] = (4, 12, 0)
objp[35] = (8, 12, 0)
objp[36] = (12, 12, 0)
objp[37] = (16, 12, 0)
objp[38] = (20, 12, 0)


objp[39] = (2, 14, 0)
objp[40] = (6, 14, 0)
objp[41] = (10, 14, 0)
objp[42] = (14, 14, 0)
objp[43] = (18, 14, 0)
###################################################################################################


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('newpattern/*.png')


count = 0
for image in images:
    #cap = cv2.VideoCapture(2)
    #num = 10
    #found = 0
    # while(found < num):  # Here, 10 can be changed to whatever number you like to choose
    # ret, img = cap.read() # Capture frame-by-frame
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = count+1
    if count == 40:
        break

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray)  # Detect blobs.

    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array(
        []), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(
        im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(
        im_with_keypoints, (4, 11), None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    # if ret == True:
    # Certainly, every loop objp is the same, in 3D.
    # beginning of loop
    objpoints.append(objp)

    # Refines the corner locations.
    corners2 = cv2.cornerSubPix(
        im_with_keypoints_gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners.
    im_with_keypoints = cv2.drawChessboardCorners(
        img, (4, 11), corners2, ret)

    # Enable the following 2 lines if you want to save the calibration images.
    # filename = str(found) +".jpg"
    # cv2.imwrite(filename, im_with_keypoints)

    #found += 1
    # end of loop

    cv2.imshow("img", im_with_keypoints)  # display
    cv2.waitKey(5000)


# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)


#  Python code to write the image (OpenCV 3.2)
#fs = cv2.FileStorage('calibration.yml', cv2.FILE_STORAGE_WRITE)
#fs.write('camera_matrix', mtx)
#fs.write('dist_coeff', dist)
# fs.release()
np.savez('C.npz', mtx=cameraMatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)
# new line above


print("Camera Calibrated:", ret)
print("\n\n")
print("Camera Matrix:\n", cameraMatrix)
print("\n\n")
print("Distortion Parameters:", dist)
print("\n\n")
print("Rotation Vectors:", rvecs)
print("\n\n")
print("Translation Vectors:", tvecs)
print("\n\n")

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
pickle.dump(dist, open("dist.pkl", "wb"))


############## UNDISTORTION #####################################################

img = cv2.imread('newpattern/img0.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (w, h), 1, (w, h))


# Undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult3.png', dst)


# Undistort with Remapping
mapx, mapy = cv2.initUndistortRectifyMap(
    cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult4.png', dst)


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))
