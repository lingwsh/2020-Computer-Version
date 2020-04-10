# import the necessary packages
import numpy as np
import imutils
import cv2
from stitcher import Stitcher


# load the two images and resize them to have a width of 400 pixels
# (for faster processing)

imageA = cv2.imread('img/1.jpg')
imageB = cv2.imread('img/2.jpg')
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)