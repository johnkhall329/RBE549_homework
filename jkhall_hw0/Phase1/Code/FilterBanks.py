import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import os

SIGMA = math.sqrt(2)

sobelx = np.array([[-1, 0 , 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobely = np.array([[1, 2 , 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

img_folder = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../Images'))

np.set_printoptions(precision=2)

def generate_oriented_DoG_filters(size, scales, orientations, sigma=SIGMA, display=False):
    bank = []

    for scale in range(1,scales+1):
        g = cv2.getGaussianKernel(size, sigma**scale)
        g = g @ g.T
        g /= np.linalg.norm(g)
        DoGx = cv2.filter2D(g, -1, sobelx)
        DoGy = cv2.filter2D(g, -1, sobely)
        thetas = np.linspace(0,2*np.pi, orientations, False)
        for theta in thetas:
            DoG = math.cos(theta)*DoGx + math.sin(theta)*DoGy

            bank.append(DoG)

    if display:
        fig, axes = plt.subplots(nrows=scales, ncols=orientations)
        fig.suptitle(f"Oriented DoG Filter Bank")
        for ax, filter in zip(axes.flatten(), bank):
            resized_filter = cv2.resize(filter, (200,200))
            ax.imshow(resized_filter, cmap='gray')
            ax.axis('off')
        fig.savefig(img_folder+'/DoG.png')
        plt.tight_layout()
        plt.show()

    return bank

def generate_LMS_and_LML_filters(size, display=False):
    LMS_bank = generate_LM_filters(size, 1, display)
    LML_bank = generate_LM_filters(size, SIGMA, display)

    if display:
        fig, axes = plt.subplots(nrows=8, ncols=12)
        fig.suptitle(f"LM Filter Bank")
        for ax, filter in zip(axes.flatten(), LMS_bank+LML_bank):
            resized_filter = cv2.resize(filter, (200,200))
            ax.imshow(resized_filter, cmap='gray')
            ax.axis('off')
        fig.savefig(img_folder+'/LM.png')
        plt.tight_layout()
        plt.show()
    return LMS_bank+LML_bank

def generate_LM_filters(size, scale=1, display=False):
    LM_bank = []
    DoG_bank = []
    L_bank = []
    G_bank = []
    sigmas = scale*np.power(SIGMA, np.arange(4))
    normalized = np.zeros((size,size))
    for sigma in sigmas[1:]:
        DoG1_bank = []
        DoG2_bank = []
        gx = cv2.getGaussianKernel(size, 3*sigma)
        gy = cv2.getGaussianKernel(size, sigma)
        G = gx @ gy.T
        G /= np.sum(G)


        thetas = np.linspace(0,np.pi, 6, False)
        for theta in thetas:
            r = cv2.getRotationMatrix2D((size//2,size//2), np.rad2deg(theta), 1.0)
            g_new = cv2.warpAffine(G, r, G.shape)
            DoG1x = cv2.filter2D(g_new, -1, sobelx)
            DoG1y = cv2.filter2D(g_new, -1, sobely)
            DoG2xx = cv2.filter2D(DoG1x, -1, sobelx)
            DoG2yy = cv2.filter2D(DoG1y, -1, sobely)
            DoG2xy = cv2.filter2D(DoG1x, -1, sobely)
            DoG1 = math.cos(theta)*DoG1x + math.sin(theta)*DoG1y
            DoG2 = math.cos(theta)**2*DoG2xx + 2*math.cos(theta)*math.sin(theta)*DoG2xy + math.sin(theta)**2*DoG2yy
            LM_bank.append(DoG1)
            LM_bank.append(DoG2)

            if display: 
                DoG1_bank.append(cv2.normalize(DoG1, normalized, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                DoG2_bank.append(cv2.normalize(DoG2, normalized, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        DoG_bank += DoG1_bank + DoG2_bank

    for sigma in np.hstack((sigmas,3*sigmas)):
        g = cv2.getGaussianKernel(size, sigma)
        G = g @ g.T
        G /= np.sum(G)
        L = cv2.Laplacian(G, -1)
        LM_bank.append(L)
        if display: L_bank.append(L)
    
    for sigma in sigmas:
        g = cv2.getGaussianKernel(size, sigma)
        G = g @ g.T
        G /= np.linalg.norm(G)
        LM_bank.append(G)
        if display: G_bank.append(G)
    
    if display:
        LM_bank = DoG_bank+L_bank+G_bank
    return LM_bank

def generate_gabor_filters(size, sigmas, lambdas, gammas, orientations, display=False):
    gabor_bank = []
    thetas = np.linspace(0,np.pi, orientations, False)
    for sigma, Lambda, gamma in zip(sigmas, lambdas, gammas):
        for theta in thetas:
            gabor_filter = gabor_kernel(size, sigma, theta, Lambda, 0.25, gamma)
            gabor_bank.append(gabor_filter)

    if display:
        fig, axes = plt.subplots(nrows=len(sigmas), ncols=orientations)
        fig.suptitle(f"Gabor Filter Bank")
        for ax, filter in zip(axes.flatten(), gabor_bank):
            resized_filter = cv2.resize(filter, (200,200))
            ax.imshow(resized_filter, cmap='gray')
            ax.axis('off')
        fig.savefig(img_folder+'/Gabor.png')
        plt.tight_layout()
        plt.show()

    return gabor_bank
    


def gabor_kernel(size, sigma, theta, Lambda, psi, gamma):
    (x,y) = np.meshgrid(np.arange(-size//2+1,size//2+1), np.arange(-size//2+1,size//2+1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 + gamma**2*y_theta**2)/sigma**2
    ) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


generate_oriented_DoG_filters(15, 2, 16, 1, False)
generate_LMS_and_LML_filters(49, False)
generate_gabor_filters(49, [5,10,15,20], [5,10,15,20], [0.75, 1,1.25, 1.25], 8, False)

normalized = np.zeros((49,49))
# g = gabor(49,SIGMA, np.pi/4, 0.25, 25, 1)
# cv2.imshow('gabor1', cv2.resize(cv2.normalize(g, normalized, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), (500,500)))
# g2 = gabor_kernel(49,5, np.pi/4, 10, 0.1, 1)
# cv2.imshow('gabor2', cv2.resize(cv2.normalize(g2, normalized, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), (500,500)))
# while True:
#     cv2.waitKey(1)
    
# generate_LM_filters(49, SIGMA, True)