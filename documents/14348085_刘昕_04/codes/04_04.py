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

def my_iFFT(FFT_image):
    image_back = np.fft.ifft2(FFT_image).real
    height, width = image_back.shape
    for i in range(height):
        for j in range(0 if i & 1 else 1, width, 2):
            image_back[i, j] = -image_back[i, j]
    return image_back

def my_Gaussian_Lowpass(image, d_0):
    fft_img = my_FFT(image)
    height, width = image.shape
    d2 = lambda u, v: (u - height // 2) * (u - height // 2) + (v - width // 2) * (v - width // 2)
    H = np.empty((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            H[i, j] = np.exp(-d2(i, j) / 2 / d_0 / d_0)
    lowpass_image = my_iFFT(H * fft_img)
    return lowpass_image

def my_Gaussian_Highpass(image, d_0):
    lowpass_img = my_Gaussian_Lowpass(image, d_0)
    return image - my_normalize(lowpass_img).astype(np.float64)

img = np.array(Image.open('../images/images_chapter_04/Fig4.11(a).jpg').convert('L'))
img1 = my_Gaussian_Highpass(img, 15)
d_0 = 56.57
img2 = my_Gaussian_Highpass(img, d_0)
ax = plt.subplot(1,3,1)
ax.set_xticks([])
ax.set_yticks([])
plt.title('original image')
plt.imshow(img, cmap='gray')
ax = plt.subplot(1,3,2)
ax.set_xticks([])
ax.set_yticks([])
plt.title('radii value of 15 with subtraction')
plt.imshow(my_normalize(img1), cmap='gray')
ax = plt.subplot(1,3,3)
ax.set_xticks([])
ax.set_yticks([])
plt.title('radii value of %.2f with subtraction' % d_0)
plt.imshow(my_normalize(img2), cmap='gray')
plt.show()
