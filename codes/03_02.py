#!/usr/bin/env python3
#coding:utf-8

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_histogram(image, bins=10):
    width, height = image.shape
    hist = [0] * 256
    for x in range(width):
        for y in range(height):
            hist[image[x,y]] += 1
    return np.array(hist)

def my_equalize_histogram(image):
    hist = my_histogram(image)
    hist = hist / image.size
    cdf = hist.cumsum()
    cdf = 255 * cdf
    cdf = np.ma.filled(cdf,0).astype(np.uint8)
    new_img = np.frompyfunc(lambda x: cdf[x], 1, 1)(image).astype(np.uint8)
    return new_img

img = np.array(Image.open('../images/images_chapter_03/Fig3.08(a).jpg').convert('L'))
plt.subplot(2,2,1)
plt.title('original image')
plt.imshow(img, cmap='gray')
hist = my_histogram(img)
plt.subplot(2,2,2)
plt.title('original histogram')
plt.plot(range(hist.shape[0]) ,hist)
equ_img = my_equalize_histogram(img)
plt.subplot(2,2,3)
plt.title('equalized image')
plt.imshow(equ_img, cmap='gray')
equ_hist = my_histogram(equ_img)
plt.subplot(2,2,4)
plt.title('equalized histogram')
plt.plot(range(equ_hist.shape[0]) ,equ_hist)
plt.show()
