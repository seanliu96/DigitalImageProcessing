#!/usr/bin/env python3
#coding:utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_histogram(image):
    height, width = image.shape
    hist = np.zeros((256), dtype=np.float64)
    for x in range(height):
        for y in range(width):
            hist[image[x,y]] += 1
    hist = hist / image.size
    return hist

def my_equalize_histogram(image):
    hist = my_histogram(image)
    cdf = hist.cumsum()
    T = 255 * cdf
    T = T.astype(np.uint8)
    new_img = np.frompyfunc(lambda x: T[x], 1, 1)(image).astype(np.uint8)
    return new_img

img = np.array(Image.open('../images/images_chapter_03/Fig3.08(a).jpg').convert('L'))
plt.subplot(2,2,1)
plt.title('original image')
plt.imshow(img, cmap='gray')
hist = my_histogram(img)
plt.subplot(2,2,2)
plt.title('original histogram')
plt.bar(range(hist.shape[0]) ,hist, color = 'g')
equ_img = my_equalize_histogram(img)
plt.subplot(2,2,3)
plt.title('equalized image')
plt.imshow(equ_img, cmap='gray')
equ_hist = my_histogram(equ_img)
plt.subplot(2,2,4)
plt.title('equalized histogram')
plt.bar(range(equ_hist.shape[0]) ,equ_hist, color = 'g')
plt.show()