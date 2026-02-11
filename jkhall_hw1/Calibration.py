import numpy as np
from scipy.optimize import least_squares

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

def projection(K, dist_coeffs, obj_p, ext):
    k1, k2 = dist_coeffs
    obj_p[:,2] = 1.0
    camera_pts = (ext@obj_p.T).T
    x = camera_pts[:,[0]]/camera_pts[:,[2]]
    y = camera_pts[:,[1]]/camera_pts[:,[2]]
    r2 = x*x + y*y
    dist = 1+k1*r2+k2*r2*r2
    camera_pts_dist = np.hstack([x*dist, y*dist, np.ones_like(x)])
    img_pred = (K@camera_pts_dist.T).T
    return img_pred

def project_points(params, obj_points, img_points, extrinsics):
    residuals=[]
    K = np.array([[params[0], 0, params[2]],
                  [0, params[1], params[3]],
                  [0,0,1]])
    k1, k2 = params[4:]
    for obj_p, img_p, ext in zip(obj_points, img_points, extrinsics):
        img_pred = projection(K, (k1,k2), obj_p, ext)
        diff = img_pred[:,:2] - img_p
        residuals.extend(diff.ravel())

    return np.array(residuals)


def optimize_params(K, obj_points, img_points, extrinsics):
    P0 = np.array([K[0,0], K[1,1], K[0,2], K[1,2], 0.0, 0.0]) # optimizing fx, fy, px, py, k1, k2 with k1,k2=0 to start
    result = least_squares(project_points, P0, args=(obj_points, img_points, extrinsics), method='lm')
    K_opt = K = np.array([[result.x[0], 0, result.x[2]],
                          [0, result.x[1], result.x[3]],
                          [0,0,1]])
    k1, k2 = result.x[4:]
    return K_opt, k1, k2