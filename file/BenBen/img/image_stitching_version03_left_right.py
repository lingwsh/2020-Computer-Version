#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:16:04 2020

@author: shuaiwang
中间，左右摇摆拼接.
从中间开始拼接，还是会有接缝问题，这个版本可以拼接五张了，结果还是不错的
"""
import numpy as np
import cv2
import glob 
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
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2BGRA)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2BGRA)
    
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
    
    img2 = img1
    img1 = result
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[t[1]:h1+t[1],t[0]:w1+t[0] ] = dst

    result = img1
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
k=num-1
mid = 0
head = 0
tail = 0
if num%2 == 0:
    mid = num/2-1
if num%2 == 1:
    mid = (num-1)/2
mid = int(mid)
if num >2:
        head = mid-1
        tail =mid+1
else:
    tail = mid+1
for i in range(num-1):
    print('stitching:'+str(i+1)+'/'+str(num-1))
        
    if i == 0:
        print('first:mid = '+str(mid))
        print('first:tail = '+str(tail))
        H = findhomograpyh(images[mid],images[tail])
        result = warpTwoImages(images[mid],images[tail],H)
        cv2.imwrite('result_08-5pics-'+str(i)+'_boat.png',result)
        cv2.imshow("img"+str(i), result)
        if tail < num-1:
            tail = tail+1
            print("tail+1: "+str(tail))
        continue
    if i%2 == 1:
        print(str(i)+'head = '+str(head))
        H = findhomograpyh(result,images[head])
        result = warpTwoImages(result,images[head],H)
        if head > 0:
            head = head-1
            print("head-1: "+str(head))
    if i%2 == 0:
        print(str(i)+'tail = '+str(tail))
        H = findhomograpyh(result, images[tail])
        result = warpTwoImages(result, images[tail],H)
        if tail < num-1:
            tail =tail+1
            print("tail+1: "+str(tail))
    cv2.imwrite('result_08-5pics-'+str(i)+'_boat.png',result)
    cv2.imshow("img"+str(i), result)
    # k = k-1

print('finished')
# cv2.imshow("img pano", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (1,5):
    cv2.waitKey(1)
