#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:16:04 2020

@author: shuaiwang
1.clean all the unnecessary codes in old files
2.add explainations
3.this code can stitch 3-4 pictures in sequence, if more than that, it will 
distort very much
"""
import numpy as np
import cv2
import glob 
import time
reprojThresh=4.0
ratio=0.75

# stitching method input two pictures and output a result of new stitching picture
def stitching(img22,img11):
    # change image into gray model
    img1 =cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    img2 =cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    # find the key points and descriptors by surf
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

# ----------------------------------------------------------------------------
# simple methods of stitching images directly. 
# It works, but will lose some pixels
def stichingimg(img22,img11,H):
    result = cv2.warpPerspective(img11, H,
                (img11.shape[1] + img22.shape[1], img11.shape[0]))
    result[0:img22.shape[0], 0:img22.shape[1]] = img22
            
    return result
# ----------------------------------------------------------------------------
def findhomograpyh(img22,img11):
    # change image into gray model
    img1 =cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
    img2 =cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
    
    # find the key points and descriptors by surf
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
    
        # compute the homography between the two sets of points with RANSAC
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
    return H
# ----------------------------------------------------------------------------
#stitching images in full pixels
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
    
# ----------------------------------------------------------------------------
#%% cylindricalWarp before image, but this code not work for this version
def cylindricalWarp(img, K):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
  
#%%
WINDOW_NAME = "Test Stitching On Mac"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
# initialtime = time.time()
cv2.startWindowThread()
print('start stitching...')
# ----------------------------------------------------------------------------
# read all images in sequence
imgs_path = glob.glob('testimg/*')
images = []
num = len(imgs_path)
ino = 1
for i in range(num):
    print('reading images: testimg/'+str(ino)+'.jpg')
    img = cv2.imread('testimg/boat'+str(ino)+'.jpg')
    images.append(img)
    ino =ino + 1
    
# ----------------------------------------------------------------------------
# use images test desc by H. stitching images from last two, then stitching 
# the result with the one before it 
k=num-2

mid = 0
head = 0
tail = 0
if num%2 == 0:
    mid = num/2-1
if num%2 == 1:
    mid = (num-1)/2
mid = int(mid)

for i in range(num-1):
    print('stitching:'+str(i+1)+'/'+str(num-1))
    if i == 0:
        H = findhomograpyh(images[i+1],images[i])
        result = warpTwoImages(images[i+1],images[i],H)
        cv2.imwrite('result_10-6pics-'+str(i)+'_boat_middle_perfect.png',result)
        cv2.imshow("img"+str(i), result)
        continue
    if i+1 <= mid:
        H = findhomograpyh(images[i+1],result)
        result = warpTwoImages(images[i+1],result,H)
        cv2.imwrite('result_10-6pics-'+str(i)+'_boat_middle_perfect.png',result)
        cv2.imshow("img"+str(i), result)
        continue
    if i+1 == mid+1 and i < num-2:
        H = findhomograpyh(images[k], images[k+1])
        result2 = warpTwoImages(images[k], images[k+1],H)
        cv2.imwrite('result_10-6pics-'+str(i)+'_boat_middle_perfect.png',result2)
        cv2.imshow("img"+str(i), result2)
        k = k-1
        continue
    if i+1 > mid+1 and i < num-2:
        H = findhomograpyh(images[k], result2)
        result2 = warpTwoImages(images[k], result2,H)
        cv2.imwrite('result_10-6pics-'+str(i)+'_boat_middle_perfect.png',result2)
        cv2.imshow("img"+str(i), result2)   
        k = k-1
        continue
    if i == num-2:
        H = findhomograpyh(result, result2)
        result = warpTwoImages(result, result2,H)
        cv2.imwrite('result_10-6pics-'+str(i)+'_boat_middle_perfect.png',result)
        cv2.imshow("img"+str(i), result)
        continue
    

print('finished')
# cv2.imshow("img pano", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (1,5):
    cv2.waitKey(1)
