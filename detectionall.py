import cv2 as cv
import numpy as np
import glob
images = glob.glob('newpattern/*.png')

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


for im in images:
    # for i in range(65):
    # is this the correct path for the image?
    #newstring = ("newpattern/img%d.png", i)
    # image = cv.imread(newstring) #was originally img3
    print("image is: \n\n")
    print(im)
    # apply gaussian blur next
    image = cv.imread(im)
    # check if these parameters make sense
    blur = cv.GaussianBlur(image, (5, 5), 0)
    # apply laplacian with 64 bit floating point
    blobs_log = cv.Laplacian(blur, cv.CV_64F)
    # the absolute function returns the absolute values of each element in the array
    # the uint8 function converts each element in the array to 8 bit integers
    blobs_log = np.uint8(np.absolute(blobs_log))

    # Detect blobs.
    keypoints = detector.detect(blobs_log)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of the blob
    im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array(
        []), (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    #cv.imshow("Keypoints", im_with_keypoints)
    cv.imshow("", im_with_keypoints)
    cv.waitKey(5000)
cv.destroyAllWindows()
