# code that counts the number of dots in an image:

# import cv2 as cv
# import glob


# images = glob.glob("newpattern/*.png")  # check if this is correct syntax

# for image in images:
#     gray = cv.imread(image, 0)
#     th, threshed = cv.threshold(
#         gray, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # not entirely sure I understand this line

#     counts = cv.findContours(threshed, cv.RETR_LIST,
#                              cv.CHAIN_APPROX_SIMPLE)[-2]  # I have no idea what this does

#     s1 = 3
#     s2 = 10
#     xcounts = []
#     for count in counts:
#         if s1 < cv.contourArea(count) and cv.contourArea(count) < s2:
#             xcounts.append(count)
#     print("The number of dots is: {}".format(len(xcounts)))


import cv2 as cv
import numpy as np


# is this the correct path for the image?
image = cv.imread('newpattern/img3.png')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# doing this because the first parameter of the threshold function has to be an image converted to grayscale


_, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


print("threshold is:\n\n")
print(threshold)
# the otsu picks a threshold using a mathematical formula
# it separates pixels into 2 classes, foreground and background


# Now we have to filter out large non-connecting objects
#contours = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# CV_RETR_EXTERNAL gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
# cv.chain_approx_simple gives only the corner points, so that less points are used and not as much storage is taken up
# contours = contours[0]  # added this line due to execution error
if len(contours) == 2:
    contours = contours[0]
else:
    contours = contours[1]

for c in contours:
    area_of_contour = cv.contourArea(c)
    # contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    if(area_of_contour < 500):
        #cv.drawContours(image, contours, -1, (0, 255, 0), 3)
        cv.drawContours(threshold, [c], 0, 0, -1)
        # -1 means drawing all the contours
        # (0, 255, 0) is a certain color: green I think
        # 3 is line thickness


# now perform morphological transformations
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

# kernel = [[0,1,0]
#             [1,1,1]
#             [0,1,0]]


# cv.morphologyEx(gray, output, kernel, iterations=3)

# this part does the actual transformation
#opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
opening = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=3)

contours = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
if len(contours) == 2:
    contours = contours[0]
else:
    contours = contours[1]
#contours = contours[0]
for c in contours:
    area = cv.contourArea(c)

    # contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    if area > 0 and area < 500:
        # this is a valid circle
        ((x, y), r) = cv.minEnclosingCircle(c)
        print("x = ")
        print(x)
        print("\n\n")
        print("y = ")
        print(y)
        print("\n\n")
        print("r = ")
        print(r)
        # finds the smallest circle that covers the contour

        # these radius and center numbers need to be ints, not floats
        cv.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 3)
        # this circles the min enclosing circle
        # in whatever color (125, 125, 125) is

cv.imshow('image', image)
cv.imshow('thresh', threshold)
cv.imshow('opening', opening)
cv.waitKey(10000)  # destorys all windows after 5 seconds
cv.destroyAllWindows
