#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import sklearn
import glob
import os
import sys
from FilterBanks import generate_oriented_DoG_filters, generate_LM_filters, generate_gabor_filters, generate_half_discs
from Gradients import brightness_map, color_map, compute_gradient, texton_map, compute_pb_lite


def main():
    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    DoG_bank = generate_oriented_DoG_filters(15, [np.sqrt(2), 2], 16, display=False)

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    LM_bank = generate_LM_filters(49, display=False)


    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    Gabor_bank = generate_gabor_filters(15, [1.75,4], [2,4], [0.9, 1.1], 8, display=False)


    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    half_discs = generate_half_discs([9,15,25],8, display=False)


    """
    Generate Texton Map
    Filter image using oriented gaussian filter bank

    Generate texture ID's using K-means clustering
    Display texton map and save image as TextonMap_ImageName.png,
    use command "cv2.imwrite('...)"
    """

    BSDS500_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../BSDS500'))
    img_dir = os.path.join(BSDS500_dir, 'Images')
    sobel_dir = os.path.join(BSDS500_dir, 'SobelBaseline')
    canny_dir = os.path.join(BSDS500_dir, 'CannyBaseline')
    images = glob.glob(img_dir+'/*.jpg')
    filter_bank = DoG_bank+LM_bank+Gabor_bank
    t_clusters = 64
    b_c_clusters = 16
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image_name = img_path.split('/')[-1] if sys.platform.startswith('linux') else img_path.split('\\')[-1]
        image_name = image_name.removesuffix('.jpg')
        t_map = texton_map(img, image_name, filter_bank, t_clusters, display=True)
        

        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Tg = compute_gradient(t_map, t_clusters, image_name, half_discs, "Tg", display=True)
        print("computed Tg")

        """
        Generate Brightness Map
        Perform brightness binning 
        """
        b_map = brightness_map(img, b_c_clusters)

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Bg = compute_gradient(b_map, b_c_clusters, image_name, half_discs, "Bg", display=True)
        print("computed Bg")

        """
        Generate Color Map
        Perform color binning or clustering
        """
        c_map = color_map(img, b_c_clusters)

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Cg = compute_gradient(c_map, b_c_clusters, image_name, half_discs, "Cg", display=True)
        print("computed Cg")

        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        sobel_baseline = cv2.imread(sobel_dir + '/' + image_name + '.png', cv2.IMREAD_GRAYSCALE)

        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        canny_baseline = cv2.imread(canny_dir + '/' + image_name + '.png', cv2.IMREAD_GRAYSCALE)

        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pb_lite = compute_pb_lite(Tg, Bg, Cg, sobel_baseline, canny_baseline, 0.5, 0.5, 5, image_name, display=True)
    
if __name__ == '__main__':
    main()
 


