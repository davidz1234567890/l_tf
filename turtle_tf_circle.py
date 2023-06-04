#!/usr/bin/env python3


# here are some new imports
import numpy as np
import cv2 as cv

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import turtlesim.msg
import tf
import rospy
import time
import roslib
roslib.load_manifest('learning_tf')


# using absolute path
with np.load('/home/huahua/catkin_ws/src/l_tf/nodes/B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    print(mtx)


criteria = (cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# now let's setup the detector
params = cv.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10  # was initially 10
params.maxThreshold = 220  # was initially 200
# Filter by Area.
params.filterByArea = True
params.minArea = 20  # originally was 100
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.025  # was originally 0.9
# Filter by Convexity
# the higher the convexity, the more circular the blob is
params.filterByConvexity = True
params.minConvexity = 0.001  # I think this was originally 0.2
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create(params)

########################################Blob Detector##############################################
# # Setup SimpleBlobDetector parameters.
# blobParams = cv.SimpleBlobDetector_Params()

# # Change thresholds
# blobParams.minThreshold = 8
# blobParams.maxThreshold = 255

# # Filter by Area.
# blobParams.filterByArea = True
# blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
# blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# # Filter by Circularity
# blobParams.filterByCircularity = True
# blobParams.minCircularity = 0.1

# # Filter by Convexity
# blobParams.filterByConvexity = True
# blobParams.minConvexity = 0.87

# # Filter by Inertia
# blobParams.filterByInertia = True
# blobParams.minInertiaRatio = 0.01

# # Create a detector with the parameters
# blobDetector = cv.SimpleBlobDetector_create(blobParams)
###################################################################################################


###################################################################################################
# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.
# objp = np.zeros((44, 3), np.float32)
# objp[0] = (0, 0, 0)
# objp[1] = (0, 72, 0)
# objp[2] = (0, 144, 0)
# objp[3] = (0, 216, 0)
# objp[4] = (36, 36, 0)
# objp[5] = (36, 108, 0)
# objp[6] = (36, 180, 0)
# objp[7] = (36, 252, 0)
# objp[8] = (72, 0, 0)
# objp[9] = (72, 72, 0)
# objp[10] = (72, 144, 0)
# objp[11] = (72, 216, 0)
# objp[12] = (108, 36,  0)
# objp[13] = (108, 108, 0)
# objp[14] = (108, 180, 0)
# objp[15] = (108, 252, 0)
# objp[16] = (144, 0, 0)
# objp[17] = (144, 72, 0)
# objp[18] = (144, 144, 0)
# objp[19] = (144, 216, 0)
# objp[20] = (180, 36, 0)
# objp[21] = (180, 108, 0)
# objp[22] = (180, 180, 0)
# objp[23] = (180, 252, 0)
# objp[24] = (216, 0, 0)
# objp[25] = (216, 72, 0)
# objp[26] = (216, 144, 0)
# objp[27] = (216, 216, 0)
# objp[28] = (252, 36, 0)
# objp[29] = (252, 108, 0)
# objp[30] = (252, 180, 0)
# objp[31] = (252, 252, 0)
# objp[32] = (288, 0, 0)
# objp[33] = (288, 72, 0)
# objp[34] = (288, 144, 0)
# objp[35] = (288, 216, 0)
# objp[36] = (324, 36, 0)
# objp[37] = (324, 108, 0)
# objp[38] = (324, 180, 0)
# objp[39] = (324, 252, 0)
# objp[40] = (360, 0, 0)
# objp[41] = (360, 72, 0)
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
# objp[1] = (0, 50, 0)
# objp[2] = (0, 100, 0)
# objp[3] = (0, 150, 0)
# objp[4] = (25, 25, 0)
# objp[5] = (25, 75, 0)
# objp[6] = (25, 125, 0)
# objp[7] = (25, 175, 0)
# objp[8] = (50, 0, 0)
# objp[9] = (50, 50, 0)
# objp[10] = (50, 100, 0)
# objp[11] = (50, 150, 0)
# objp[12] = (75, 25,  0)
# objp[13] = (75, 75, 0)
# objp[14] = (75, 125, 0)
# objp[15] = (75, 175, 0)
# objp[16] = (100, 0, 0)
# objp[17] = (100, 50, 0)
# objp[18] = (100, 100, 0)
# objp[19] = (100, 150, 0)
# objp[20] = (125, 25, 0)
# objp[21] = (125, 75, 0)
# objp[22] = (125, 125, 0)
# objp[23] = (125, 175, 0)
# objp[24] = (150, 0, 0)
# objp[25] = (150, 50, 0)
# objp[26] = (150, 100, 0)
# objp[27] = (150, 150, 0)
# objp[28] = (175, 25, 0)
# objp[29] = (175, 75, 0)
# objp[30] = (175, 125, 0)
# objp[31] = (175, 175, 0)
# objp[32] = (200, 0, 0)
# objp[33] = (200, 50, 0)
# objp[34] = (200, 100, 0)
# objp[35] = (200, 150, 0)
# objp[36] = (225, 25, 0)
# objp[37] = (225, 75, 0)
# objp[38] = (225, 125, 0)
# objp[39] = (225, 175, 0)
# objp[40] = (250, 0, 0)
# objp[41] = (250, 50, 0)
# objp[42] = (250, 100, 0)
# objp[43] = (250, 150, 0)

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

# objp = np.zeros((7*10, 3), np.float32)
# objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)
# print(objp)
# axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(
        tuple(imgpts[0].ravel())[0]), int(tuple(imgpts[0].ravel())[1])), (255, 0, 0), 5)
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(
        tuple(imgpts[1].ravel())[0]), int(tuple(imgpts[1].ravel())[1])), (0, 255, 0), 5)
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(
        tuple(imgpts[2].ravel())[0]), int(tuple(imgpts[2].ravel())[1])), (0, 0, 255), 5)
    return img


def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin
# """
# Helper function to make an array of random numbers having shape (n, )
# with each number distributed Uniform(vmin, vmax).
# """


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

np.random.seed(19680801)


fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax = fig.add_subplot(1, 2, 1, projection='3d')
ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot()


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

pts = []


if __name__ == '__main__':
    rospy.init_node('turtle1_tf_broadcaster')
    turtlename = rospy.get_param('~turtle')

    print("!!!")
    print('/%s/pose' % turtlename)

    # rospy.Subscriber('/%s/pose' % turtlename,
    #                  turtlesim.msg.Pose,
    #                  handle_turtle_pose,
    #                  turtlename)
    #print("function is being called")

    br = tf.TransformBroadcaster()
    # br.sendTransform((0, 0, 0),
    #                  tf.transformations.quaternion_from_euler(0, 0, 0),
    #                  rospy.Time.now(),
    #                  turtlename,
    #                  "map")
    # time.sleep(5)
    # br.sendTransform((1, 1, 0),
    #                  tf.transformations.quaternion_from_euler(0, 0, 0),
    #                  rospy.Time.now(),
    #                  turtlename,
    #                  "map")
    # time.sleep(5)
    # br.sendTransform((0, 0, 0),
    #                  tf.transformations.quaternion_from_euler(0, 0, 0),
    #                  rospy.Time.now(),
    #                  turtlename,
    #                  "map")
    #i = 0
    while(True):
        # br.sendTransform((i, i, 0),
        #                  tf.transformations.quaternion_from_euler(0, 0, 0),
        #                  rospy.Time.now(),
        #                  turtlename,
        #                  "map")
        # time.sleep(0.5)
        # i += 0.1
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # newline
        blur = cv.GaussianBlur(frame, (5, 5), 0)

        blobs_log = cv.Laplacian(blur, cv.CV_64F)
    # the absolute function returns the absolute values of each element in the array
    # the uint8 function converts each element in the array to 8 bit integers
        blobs_log = np.uint8(np.absolute(blobs_log))

        # Display the resulting frame
        #cv.imshow('frame', gray)

        keypoints = detector.detect(blobs_log)

        #keypoints = blobDetector.detect(gray)

        im_with_keypoints = cv.drawKeypoints(frame, keypoints, np.array(
            []), (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        im_with_keypoints_gray = cv.cvtColor(
            im_with_keypoints, cv.COLOR_BGR2GRAY)
        rett, corners = cv.findCirclesGrid(
            im_with_keypoints, (4, 11), None, flags=cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid
        #i = 0

        # corners2 = cv.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
        # imgpoints.append(corners2)

        # # Draw and display the corners.
        # im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        # success, corners = cv.findChessboardCorners(
        #     gray, (10, 7), None)  # new line

        if rett == True:  # new conditional
            cv.imshow('img', frame)
            cv.waitKey(1)
            # corners2 = cv.cornerSubPix(
            #     gray, corners, (11, 11), (-1, -1), criteria)

            corners2 = cv.cornerSubPix(
                im_with_keypoints_gray, corners, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors.
            # print(corners2)
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            br.sendTransform((tvecs[0]/1000, tvecs[1]/1000, tvecs[2]/1000),
                             tf.transformations.quaternion_from_euler(
                                 rvecs[0], rvecs[1], rvecs[2]),
                             rospy.Time.now(),
                             turtlename,
                             "map")
            # time.sleep(0.5)
            pts.append(tvecs)
            print(len(pts))
            print(rvecs)
            # if (len(pts) == 150):
            #     break
            #distance = np.linalg.norm(tvecs)

        else:
            cv.imshow('img', frame)
            cv.waitKey(1)

        if cv.waitKey(1) == ord('q'):
            break

    rospy.spin()
