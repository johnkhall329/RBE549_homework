import cv2
import numpy as np
# from scipy import optimize
import glob
from Homography import compute_homography, format_V
from Calibration import extract_intrinsics, extract_extrinsics, optimize_params

def wrapper():
    calib_folder = 'Calibration_Imgs'
    calib_img_paths = glob.glob(calib_folder+'/*.jpg')
    pattern_shape = (9,6)
    size = 0.0215

    obj_points = np.zeros((pattern_shape[0]*pattern_shape[1],3), np.float32)
    obj_points[:,:2] = np.mgrid[0:pattern_shape[0],0:pattern_shape[1]].T.reshape(-1,2)*size

    vs = []
    hs = []
    obj_p = []
    img_p = []
    img_ps = []
    for i, calib_img_path in enumerate(calib_img_paths):
        img = cv2.imread(calib_img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img, pattern_shape, None)
        if ret:
            obj_p.append(obj_points)
            img_ps.append(corners)
            img_p.append(corners.reshape(-1,2))
            H_i = compute_homography(obj_points, corners.reshape(-1,2))
            hs.append(H_i)
            V_i = format_V(H_i)
            vs.append(V_i)
    
    V = np.vstack(vs)
    b = np.linalg.svd(V)[2][-1]
    K, lamda = extract_intrinsics(b)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_ps, img.shape[::-1], None, None)
    extrinsics = [extract_extrinsics(H, K, lamda) for H in hs]
    K, k1, k2 = optimize_params(K, obj_p, img_p, extrinsics)
    print(K, k1, k2)
            

if __name__ == '__main__':
    wrapper()