#!/usr/bin/env python3
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
    image_back = np.fft.ifft2(FFT_image)
    height, width = image_back.shape
    for i in range(height):
        for j in range(0 if i & 1 else 1, width, 2):
            image_back[i, j] = -image_back[i, j]
    return image_back