import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import cv2

img_folder = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../Images'))
BSDS500_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../BSDS500'))
sobel_dir = os.path.join(BSDS500_dir, 'SobelBaseline')
canny_dir = os.path.join(BSDS500_dir, 'CannyBaseline')

texton_maps = glob.glob(img_folder + '/TextonMap_*.png')
texton_maps.append(texton_maps.pop(1))

brightness_maps = glob.glob(img_folder + '/BrightnessMap_*.png')
brightness_maps.append(brightness_maps.pop(1))

color_maps = glob.glob(img_folder + '/ColorMap_*.png')
color_maps.append(color_maps.pop(1))

texton_gradients = glob.glob(img_folder + '/Tg_*.png')
texton_gradients.append(texton_gradients.pop(1))

brightness_gradients = glob.glob(img_folder + '/Bg_*.png')
brightness_gradients.append(brightness_gradients.pop(1))

color_gradients = glob.glob(img_folder + '/Cg_*.png')
color_gradients.append(color_gradients.pop(1))

sobel_inputs = glob.glob(sobel_dir + '/*.png')
sobel_inputs.append(sobel_inputs.pop(1))

canny_inputs = glob.glob(canny_dir + '/*.png')
canny_inputs.append(canny_inputs.pop(1))

pb_lite_maps = glob.glob(img_folder + '/PbLite_*.png')
pb_lite_maps.append(pb_lite_maps.pop(1))

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,10))
fig.suptitle('Pb-Lite Outputs')
for ax, map in zip(axes.flatten(), pb_lite_maps):
    im = plt.imread(map)
    ax.imshow(im)
    ax.axis('off')
    
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,10))
fig.suptitle('Sobel Inputs')
for ax, map in zip(axes.flatten(), sobel_inputs):
    im = cv2.imread(map, cv2.IMREAD_GRAYSCALE)
    ax.imshow(im, cmap='gray')
    ax.axis('off')

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,10))
fig.suptitle('Canny Inputs')
for ax, map in zip(axes.flatten(), canny_inputs):
    im = cv2.imread(map, cv2.IMREAD_GRAYSCALE)
    ax.imshow(im, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()