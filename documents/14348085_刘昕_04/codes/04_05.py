#/usr/bin/env python3
#coding:utf-8

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def my_FFT(image):
    height, width = image.shape
    FFT_image = np.empty((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            coff = -1 if (i+j) & 1 else 1
            FFT_image[i, j] = coff * image[i, j]
    return np.fft.fft2(FFT_image)

def my_iFFT(FFT_image):
    image_back = np.fft.ifft2(FFT_image).real
    height, width = image_back.shape
    for i in range(height):
        for j in range(0 if i & 1 else 1, width, 2):
            image_back[i, j] = -image_back[i, j]
    return image_back

def my_padding(image1, image2):
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    height = height1 + height2 - 1
    width = width1 + width2 - 1
    padded_image1 = np.zeros((height, width), dtype=np.uint8)
    padded_image2 = np.zeros((height, width), dtype=np.uint8)
    for i in range(height1):
        for j in range(width1):
            padded_image1[i,j] = image1[i,j]
    for i in range(height2):
        for j in range(width2):
            padded_image2[i,j] = image2[i,j]
    return padded_image1, padded_image2

def my_coordination(image1, image2):
    fft_image1 = my_FFT(image1)
    fft_image2_conj = my_FFT(image2).conj()
    coordinating_image = fft_image1 * fft_image2_conj
    return my_iFFT(coordinating_image)

original_img = np.array(Image.open('../images/images_chapter_04/Fig4.41(a).jpg').convert('L'))
template_img = np.array(Image.open('../images/images_chapter_04/Fig4.41(b).jpg').convert('L'))
padded_original_img, padded_template_img = my_padding(original_img, template_img)
coordinating_img = my_coordination(padded_original_img, padded_template_img)
position = np.argmax(coordinating_img)
height, width = coordinating_img.shape
x = position // height
y = position % width
print(x, y)
ax = plt.subplot(1,3,1)
ax.set_xticks([])
ax.set_yticks([])
plt.title('original image')
plt.imshow(original_img, cmap='gray')
ax = plt.subplot(1,3,2)
ax.set_xticks([])
ax.set_yticks([])
plt.title('template image')
plt.imshow(template_img, cmap='gray')
ax = plt.subplot(1,3,3)
ax.set_xticks([])
ax.set_yticks([])
plt.title('coordination')
plt.imshow(coordinating_img, cmap='gray')
plt.show()
