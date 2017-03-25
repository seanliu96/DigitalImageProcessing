#!/usr/bin/env python3
#coding:utf-8

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_scale(image):
    scaled_image = image.copy()
    scaled_image = (scaled_image - scaled_image.min()) * 255 / (scaled_image.max() - scaled_image.min())
    return scaled_image.astype(np.uint8)

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


img = np.array(Image.open('../images/images_chapter_03/Fig3.40(a).jpg').convert('L'))
laplacian_mask = np.array([ [-1, -1, -1], 
                            [-1, 8, -1], 
                            [-1, -1, -1]])
filtered_img = my_spatial_masking(img, laplacian_mask)
scaled_filtered_img = my_scale(filtered_img)
new_img = img + filtered_img

plt.subplot(2,2,1)
plt.title('original image')
plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.title('laplacian-filtered image')
plt.imshow(filtered_img, cmap='gray')
plt.subplot(2,2,3)
plt.title('scaled laplacian-filtered-image')
plt.imshow(scaled_filtered_img, cmap='gray')
plt.subplot(2,2,4)
plt.title('final image')
plt.imshow(new_img, cmap='gray')
plt.show()