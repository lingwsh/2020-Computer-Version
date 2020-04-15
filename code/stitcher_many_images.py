'''use the opencv implenmented image stitcher'''

import numpy as np
import cv2
import glob 
import imutils


imgs_path = glob.glob('images/*')
images = []

for fname in imgs_path:
    img = cv2.imread(fname)
    images.append(img)

print("[INFO] stitching images...")
stitcher = cv2.createStitcher() 
(status, stitched) = stitcher.stitch(images)
stitched = imutils.resize(stitched, width=1024)

if status == 0:
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

# The stitching failed, likely due to not enough keypoints being detected
else:
    print("[INFO] image stitching failed ({})".format(status))