#!/usr/bin/env python3
#coding:utf-8

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_normalize(image):
    scaled_image = np.frompyfunc(lambda x: max(0, min(x, 255)), 1, 1)(image).astype(np.uint8)
    return scaled_image

def my_scale(image):
    scaled_image = image.copy()
    scaled_image = (scaled_image - scaled_image.min()) * 255 / (scaled_image.max() - scaled_image.min())
    return scaled_image.astype(np.uint8)

def my_spatial_masking(image, mask):
    filtered_image = np.zeros(image.shape, dtype=np.float64)
    height, width = image.shape
    h, w = mask.shape
    mid_w = w // 2
    mid_h = h // 2
    for x in range(height):
        for y in range(width):
            for i in range(h):
                for j in range(w):
                    px = x + i - mid_h
                    py = y + j - mid_w
                    if px >= 0 and px < height and py >= 0 and py < width:
                        filtered_image[x, y] += mask[i,j] * image[px, py]
    return filtered_image


img = np.array(Image.open('../images/images_chapter_03/Fig3.40(a).jpg').convert('L'))
laplacian_mask = np.array([ [-1, -1, -1], 
                            [-1, 8, -1], 
                            [-1, -1, -1]])
filtered_img = my_spatial_masking(img, laplacian_mask)
new_img = my_normalize(img + filtered_img)
plt.subplot(2,2,1)
plt.title('original image')
plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.title('laplacian-filtered image')
plt.imshow(my_normalize(filtered_img), cmap='gray')
plt.subplot(2,2,3)
plt.title('scaled laplacian-filtered-image')
plt.imshow(my_scale(filtered_img), cmap='gray')
plt.subplot(2,2,4)
plt.title('final image')
plt.imshow(my_normalize(new_img), cmap='gray')
plt.show()