#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:16:04 2020

@author: shuaiwang
1.in this version2, it will test how to produce correct cylindrical images
2.when it stitchs the result images, how to solve the transparent problem. 
Sometimes, the images will lost transparent pixels in the cylindrical progress and 
the transparent pixels will lost transparent in the stitching progress.
"""
import numpy as np
import cv2
import glob 
# import time
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

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin),borderMode=cv2.BORDER_TRANSPARENT)
    
    
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
    # result = overlay_transparent(result,img1,t[1],t[0])
    # result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
# ----------------------------------------------------------------------------

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
imgs_path = glob.glob('testimg/*')
images = []
num = len(imgs_path)
ino = 1
for i in range(num):
    print('reading images: testimg/'+str(ino)+'.jpg')
    img = cv2.imread('testimg/boat'+str(ino)+'.jpg')
    h, w = img.shape[:2]
    K = np.array([[3000,0,w/2],[0,2000,h/2],[0,0,1]]) # mock intrinsics
    img = cylindricalWarp(img, K)
    images.append(img)
    ino =ino + 1
    
    
# mm=0
# for img in images2:
#     h, w = img.shape[:2]
#     K = np.array([[3000,0,w/2],[0,3000,h/2],[0,0,1]]) # mock intrinsics
#     img_cyl = cylindricalWarp(img, K)
#     cv2.imwrite(str(mm)+"result.png", img_cyl)
#     mm = mm+1


#%%
WINDOW_NAME = "Test Stitching On Mac"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
# initialtime = time.time()
cv2.startWindowThread()
print('start stitching...')
# ----------------------------------------------------------------------------
# read all images in sequence
# imgs_path = glob.glob('testimg/*')
# images = []
# num = len(imgs_path)
# ino = 1
# for i in range(num):
#     print('reading images: testimg/'+str(ino)+'.jpg')
#     img = cv2.imread('testimg/'+str(ino)+'.png')
#     images.append(img)
#     ino =ino + 1
    
# ----------------------------------------------------------------------------
# use images test desc by H. stitching images from last two, then stitching 
# the result with the one before it 
k=num-2
for i in range(num-1):
    print('stitching:'+str(i+1)+'/'+str(num-1))
    if i == 0:
        # cv2.imshow("imgk+1", images[k+1])
        # cv2.imshow("imgk", images[k])
        H = findhomograpyh(images[k],images[k+1])
        result = warpTwoImages(images[k],images[k+1],H)
        # cv2.imwrite('result_07-4pics-'+str(k)+'_room.png',result)
        cv2.imshow("img"+str(k), result)
        k = k-1
        continue
    H = findhomograpyh(images[k],result)
    result = warpTwoImages(images[k],result,H)
    # cv2.imwrite('result_07-4pics-'+str(k)+'_room.png',result)
    cv2.imshow("img"+str(k), result)
    k = k-1

print('finished')
# cv2.imshow("img pano", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (1,5):
    cv2.waitKey(1)
