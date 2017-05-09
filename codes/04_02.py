#/usr/bin/env python3
#coding:utf-8

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_normalize(image):
    scaled_image = np.frompyfunc(lambda x: max(0, min(x, 255)), 1, 1)(image).astype(np.uint8)
    return scaled_image

def my_FFT(image):
    height, width = image.shape
    FFT_image = np.empty((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            coff = -1 if (i+j) & 1 else 1
            FFT_image[i, j] = coff * image[i, j]
    return np.fft.fft2(FFT_image)

img = np.array(Image.open('../images/images_chapter_04/Fig4.11(a).jpg').convert('L'))
fft_img = my_FFT(img)
ax = plt.subplot(2,1,1)
ax.set_xticks([])
ax.set_yticks([])
plt.title('origin image')
plt.imshow(img, cmap='gray')
ax = plt.subplot(2,1,2)
ax.set_xticks([])
ax.set_yticks([])
plt.title('spectrum')
plt.imshow(np.log(np.abs(fft_img)), cmap='gray')
plt.show()
height, weight = fft_img.shape
print('The average value is', fft_img[height // 2, weight // 2].real / height / weight)