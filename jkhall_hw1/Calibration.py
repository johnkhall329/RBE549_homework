import numpy as np
from scipy.optimize import least_squares
from Projection import *

def extract_intrinsics(b):
    if b[0]<0:
        b = -b
    b11, b12, b22, b13, b23, b33 = b
    py = (b12*b13-b11*b23)/(b11*b22-b12**2)
    lamda = b33 - (b13**2 + py*(b12*b13-b11*b23))/b11
    fx = np.sqrt(lamda/b11)
    fy = np.sqrt(lamda*b11/(b11*b22-b12**2))
    px = -b13*fx**2/lamda

    K = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0,0,1]], dtype=np.float32)
    
    return K, lamda

def extract_extrinsics(H, K, lamda):
    K_inv = np.linalg.inv(K)
    r1 = lamda*K_inv@H[:,0]
    r2 = lamda*K_inv@H[:,1]
    # r3 = np.cross(r1, r2)
    t = lamda*K_inv@H[:,2]
    ext = np.array([r1,r2,t]).T
    return ext

def optimize_params(K, obj_points, img_points, extrinsics):
    P0 = np.array([K[0,0], K[1,1], K[0,2], K[1,2], 0.0, 0.0]) # optimizing fx, fy, px, py, k1, k2 with k1,k2=0 to start
    result = least_squares(project_points, P0, args=(obj_points, img_points, extrinsics), method='lm')
    K_opt = K = np.array([[result.x[0], 0, result.x[2]],
                          [0, result.x[1], result.x[3]],
                          [0,0,1]])
    k1, k2 = result.x[4:]
    return K_opt, k1, k2