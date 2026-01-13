import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import os
from sklearn.cluster import k_means

img_folder = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../Images'))

def texton_map(img, image_name, filter_bank, n_clusters=64, display=False):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texton_img = np.zeros((img.shape[0], img.shape[1], len(filter_bank)))

    for i,filter in enumerate(filter_bank):
        filtered_img = cv2.filter2D(gray_img, -1, filter)
        texton_img[:,:,i] = filtered_img
    print("starting texton k-means")
    _, texton_map, _ = k_means(texton_img.reshape((img.shape[0]*img.shape[1], len(filter_bank))),n_clusters) 
    print("completed texton k-means")
    texton_map = texton_map.reshape(gray_img.shape)
    if display:
        im = plt.imshow(texton_map,cmap='jet')
        plt.colorbar(im, label="Cluster")
        plt.title(f"Texton Map of Image {image_name}")
        plt.axis('off')
        plt.savefig(img_folder+"/TextonMap_"+image_name)
        plt.show()

    return texton_map

def brightness_map(img, n_clusters=16):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, brightness_map, _ = k_means(gray_img.reshape(-1,1),n_clusters)

    brightness_map = brightness_map.reshape(gray_img.shape)
    return brightness_map

def color_map(img, n_clusters=16):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, color_map, _ = k_means(hsv_img.reshape((img.shape[0]*img.shape[1], 3)),n_clusters)

    color_map = color_map.reshape(img.shape[:2])
    return color_map

def compute_gradient(map, n_clusters, image_name, masks, gradient_name, display=False):
    full_gradient = np.zeros((map.shape[0], map.shape[1], len(masks)//2))
    for i in range(0, len(masks), 2):
        chi_sq = np.zeros((map.shape[0], map.shape[1]))
        for j in range(n_clusters):
            tmp = (map == j).astype(np.float32)
            mask_left = masks[i]
            mask_right = masks[i+1] 

            g_i = cv2.filter2D(tmp, -1, mask_left)
            h_i = cv2.filter2D(tmp, -1, mask_right)
            chi_sq += 0.5* ((g_i - h_i)**2) / (g_i + h_i + 1e-10) # small constant to avoid division by 0
        full_gradient[:,:,i//2] = chi_sq
    gradient = np.mean(full_gradient, axis=2)

    if display:
        im = plt.imshow(gradient,cmap='jet')
        plt.colorbar(im, label="Mean Gradient Magnitude")
        plt.title(f"{gradient_name} of Image {image_name}")
        plt.axis('off')
        plt.savefig(img_folder+"/"+gradient_name+"_"+image_name+".png")
        plt.show()

    return gradient

def compute_pb_lite(Tg, Bg, Cg, sobel, canny, w_sobel, w_canny, img_name, display=False):
    feature_weight = (Tg+Bg+Cg)/3
    blank = np.zeros_like(feature_weight)
    # feature_weight = cv2.normalize(feature_weight, blank, 0, 255, cv2.NORM_MINMAX)
    pb_lite = feature_weight*(w_sobel*sobel + w_canny*canny)
    pb_lite = cv2.inRange(cv2.normalize(pb_lite, blank, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 1, 255)
    if display:
        im = plt.imshow(pb_lite,cmap='gray')
        # plt.colorbar(im, label="Mean Gradient Magnitude")
        plt.title(f"Pb-lite of Image {img_name}")
        plt.savefig(img_folder+"/PbLite_"+img_name+".png")
        plt.show()
    return pb_lite