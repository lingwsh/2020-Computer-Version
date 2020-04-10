#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:17:20 2020

@author: shuaiwang
"""

import cv2 as cv
import numpy as np
img = cv.imread('Test3.png',0)
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
#%%
img2 = cv.imread('Test1.png',0)
cv.namedWindow('image2', cv.WINDOW_NORMAL)
cv.imshow('image2',img2)
cv.waitKey(0)
cv.destroyAllWindows()