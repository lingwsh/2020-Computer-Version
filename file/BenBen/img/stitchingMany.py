#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:34:04 2020

@author: shuaiwang
"""
import numpy as np
import cv2
import glob 
import time
reprojThresh=4.0
ratio=0.75
def stitching(img22,img11):
    
    
    img1 =cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    img2 =cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    
    surf = cv2.xfeatures2d.SURF_create()
    
    # keypoints_sift, descriptors = sift.detectAndCompute(img, None)
    keypoints_surf11, descriptors1 = surf.detectAndCompute(img1, None)
    # keypoints_orb, descriptors = orb.detectAndCompute(img, None)
    keypoints_surf22, descriptors2 = surf.detectAndCompute(img2, None)
    
    keypoints_surf1 = np.float32([kp.pt for kp in keypoints_surf11])
    keypoints_surf2 = np.float32([kp.pt for kp in keypoints_surf22])
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    
    # Apply ratio test
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good.append((m[0].trainIdx, m[0].queryIdx))
            
    # computing a homography requires at least 4 matches
    if len(good) > 4:
        # construct the two sets of points
        ptsA = np.float32([keypoints_surf1[i] for (_, i) in good])
        ptsB = np.float32([keypoints_surf2[i] for (i, _) in good])
    
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
    
    
    result = cv2.warpPerspective(img11, H,
                (img11.shape[1] + img22.shape[1], img11.shape[0]))
    # cv2.imshow("Result0", result)
    result[0:img22.shape[0], 0:img22.shape[1]] = img22
            
    return result
        
def findhomograpyh(img22,img11):
    
    img1 =cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    img2 =cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    
    surf = cv2.xfeatures2d.SURF_create()
    
    # keypoints_sift, descriptors = sift.detectAndCompute(img, None)
    keypoints_surf11, descriptors1 = surf.detectAndCompute(img1, None)
    # keypoints_orb, descriptors = orb.detectAndCompute(img, None)
    keypoints_surf22, descriptors2 = surf.detectAndCompute(img2, None)
    
    keypoints_surf1 = np.float32([kp.pt for kp in keypoints_surf11])
    keypoints_surf2 = np.float32([kp.pt for kp in keypoints_surf22])
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    
    # Apply ratio test
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good.append((m[0].trainIdx, m[0].queryIdx))
            
    # computing a homography requires at least 4 matches
    if len(good) > 4:
        # construct the two sets of points
        ptsA = np.float32([keypoints_surf1[i] for (_, i) in good])
        ptsB = np.float32([keypoints_surf2[i] for (i, _) in good])
    
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
    return H

    
    # -------------------------------------
def stichingimg(img22,img11,H):
    result = cv2.warpPerspective(img11, H,
                (img11.shape[1] + img22.shape[1], img11.shape[0]))
    # cv2.imshow("Result0", result)
    result[0:img22.shape[0], 0:img22.shape[1]] = img22
            
    return result

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result
    
    
WINDOW_NAME = "Test Stitching On Mac"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
# initialtime = time.time()
cv2.startWindowThread()
print('start stitching...')
# ----------------------------------------------------------------------------
imgs_path = glob.glob('testimg/*')
images = []
num = len(imgs_path)
ino = 1
for i in range(num):
    print('reading images: testimg/'+str(ino)+'.jpg')
    img = cv2.imread('testimg/boat'+str(ino)+'.jpg')
    # cv2.imshow("img"+str(i), img)
    images.append(img)
    ino =ino + 1
    
# H = []
# for i in range(num-1):
#     print('find H:'+str(i+1)+'/'+str(num-1))
#     # if i == num-1:
#     #     break
#     hh = findhomograpyh(images[i],images[i+1])
#     H.append(hh)
#%%
# k=num-2
# for i in range(num-1):
#     print('stitching:'+str(i+1)+'/'+str(num-1))
#     if i == 0:
#         result = warpTwoImages(images[k],images[k+1],H[k])
#         continue

#         # result = stichingimg(images[k],images[k+1],H[k])
#         # warpTwoImages(img1, img2, H)
#     result = warpTwoImages(images[k],result,H[k])
#     # result = stichingimg(images[k],result,H[k])
#     cv2.imshow("img"+str(k), result)
#     k = k-1
# ----------------------------------------------------------------------------
# use different images test desc by H
k=num-2
print(k)
for i in range(num-1):
    print('stitching:'+str(i+1)+'/'+str(num-1))
    if i == 0:
        H = findhomograpyh(images[k],images[k+1])
        result = warpTwoImages(images[k],images[k+1],H)
        print(k)
        k = k-1
        continue

        # result = stichingimg(images[k],images[k+1],H[k])
        # warpTwoImages(img1, img2, H)
    H = findhomograpyh(images[k],result)
    result = warpTwoImages(images[k],result,H)
    print(k)
    cv2.imwrite('result_04-'+str(k)+'_boat.png',result)
    cv2.imshow("img"+str(k), result)
    k = k-1
# ----------------------------------------------------------------------------
# result
# for i in range(num):
#     print('stitching'+str(i+1)+'/'+str(num))
#     if i == num-1:
#         break
#     if i == 0:
#         result = stitching(images[i],images[i+1])
#     result = stitching(result,images[i+1])
# ----------------------------------------------------------------------------
# img11 =cv2.imread('1.jpg', cv2.IMREAD_COLOR)
# img22 =cv2.imread('2.jpg', cv2.IMREAD_COLOR)
# imgresult = stitching(img11,img22)

# ----------------------------------------------------------------------------
#%%
# result1 = warpTwoImages(images[2],images[3],H[2])
# cv2.imshow("img3-4", result1)
# result2 = warpTwoImages(images[1],images[2],H[1])
# cv2.imshow("img2-3-4", result2)
# result = warpTwoImages(images[0],images[1],H[0])
#%%
# print('stitching 1/3')
# h1 = findhomograpyh(images[2],images[3])
# result1 = warpTwoImages(images[2],images[3],h1)
# cv2.imwrite('result_03-1_myroom3-4.png',result1)
# cv2.imshow("img3-4", result1)


# print('stitching 2/3')
# h2 = findhomograpyh(images[1],result1)
# result2 = warpTwoImages(images[1],result1,h2)
# cv2.imwrite('result_03-2_myroom2-3-4.png',result2)
# cv2.imshow("img2-3-4", result2)

# print('stitching 3/3')
# h3 = findhomograpyh(images[0],result2)
# result3 = warpTwoImages(images[0],result2,h3)
# cv2.imwrite('result_03-3_myroom1-2-3-4.png',result3)
# cv2.imshow("img1-2-3-4", result3)
# print('finishing stitching and show result')

# img1 = cv2.imread('testimg/1.jpg')
# img2 = cv2.imread('testimg/2.png')
# img3 = cv2.imread('testimg/222.jpg')
# img4 = cv2.imread('testimg/2.jpg')
# hhh = findhomograpyh(img1,img3)
# result3 = warpTwoImages(img1,img3,hhh)
# cv2.imwrite('result_03-3-5_myroom1-2-3-4.png',result3)
print('finished')
cv2.imshow("img pano", result)
# cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (1,5):
    cv2.waitKey(1)
