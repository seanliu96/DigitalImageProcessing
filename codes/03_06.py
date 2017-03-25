#!/usr/bin/env python3
#coding:utf-8

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_spatial_masking(image, mask):
    filtered_image = np.zeros(image.shape, dtype=np.int32)
    width, height = image.shape
    w, h = mask.shape
    mid_w = w // 2
    mid_h = h // 2
    for x in range(width):
        for y in range(height):
            for i in range(w):
                for j in range(h):
                    px = x + i - mid_w
                    py = y + j - mid_h
                    if px >= 0 and px < width \
                    and py >= 0 and py < height:
                        filtered_image[x,y] += mask[i,j] * image[px, py]
    return filtered_image


img = np.array(Image.open('../images/images_chapter_03/Fig3.43(a).jpg').convert('L'))
laplacian_mask = np.array([ [-1, -1, -1], 
                            [-1, 8, -1], 
                            [-1, -1, -1]])
filtered_img = my_spatial_masking(img, laplacian_mask)
new_img_A_0 = filtered_img
new_img_A_1 = img + filtered_img
new_img_A_1_7 = 1.7 * img + filtered_img

plt.subplot(2,2,1)
plt.title('original image')
plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.title('A = 0 image')
plt.imshow(new_img_A_0, cmap='gray')
plt.subplot(2,2,3)
plt.title('A = 1 image')
plt.imshow(new_img_A_1, cmap='gray')
plt.subplot(2,2,4)
plt.title('A = 1.7 image')
plt.imshow(new_img_A_1_7, cmap='gray')
plt.show()