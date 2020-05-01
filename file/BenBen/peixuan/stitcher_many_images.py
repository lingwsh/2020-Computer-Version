'''use the opencv implenmented image stitcher'''

import numpy as np
import cv2
import glob 
import imutils

WINDOW_NAME = "Test Stitching On Mac"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
# initialtime = time.time()
cv2.startWindowThread()

imgs_path = glob.glob('img/*')
images = []
num = len(imgs_path)
ino = 1
for i in range(num):
    print(str(ino)+'.jpg')
    img = cv2.imread('img/'+str(ino+1)+'.jpg')
    # cv2.imshow("img"+str(i+1), img)
    images.append(img)
    ino =ino + 1
# mm = 0
# for ii in images:
#     cv2.imshow("img"+str(mm), ii)
#     mm = mm+1

print("[INFO] stitching images...")
stitcher = cv2.Stitcher.create() 
# stitcher = cv2.createStitcher() 
(status, stitched) = stitcher.stitch(images)
stitched = imutils.resize(stitched, width=1024)

if status == 0:
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

# The stitching failed, likely due to not enough keypoints being detected
else:
    print("[INFO] image stitching failed ({})".format(status))
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)