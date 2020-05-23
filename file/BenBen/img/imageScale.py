#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:45:42 2020

@author: shuaiwang
"""
import cv2

def scaleImages(images, fixmax):
    scaleImages = []
    for img in images:
        w = img.shape[1]
        h = img.shape[0]
        scale = 0
        # caculate the scale by fixmax of the max height or width
        if h>w and h>fixmax:
            scale = int(fixmax/h*100)
        elif w>h and w>fixmax:
            scale = int(fixmax/w*100)
        elif h<fixmax and w<fixmax:
            scale = 100
         
        width = int(img.shape[1] * scale / 100)
        height = int(img.shape[0] * scale / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ',img.shape+'=>'+resized.shape)
        scaleImages.append(resized)
        return scaleImages,scale
    
fixmax = 1000
WINDOW_NAME = "Test Stitching On Mac"
cv2.namedWindow(WINDOW_NAME)
cv2.startWindowThread()
 
img = cv2.imread('testing2/1.jpg')
cv2.imwrite('testvh.png',img)
print('Original Dimensions : ',img.shape)
w = img.shape[1]
h = img.shape[0]
scale = 0
if h>w and h>fixmax:
    scale = int(fixmax/h*100)
elif w>h and w>fixmax:
    scale = int(fixmax/w*100)
elif h<fixmax and w<fixmax:
    scale = 100
 
width = int(img.shape[1] * scale / 100)
height = int(img.shape[0] * scale / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
print('finished')
# cv2.imshow("img pano", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (1,5):
    cv2.waitKey(1)

#%%
vidcap = cv2.VideoCapture('testing4-video/build-h.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)