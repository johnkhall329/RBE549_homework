import cv2
import numpy as np
# from scipy import optimize
import glob
import matplotlib.pyplot as plt
from Homography import compute_homography, format_V
from Calibration import extract_intrinsics, extract_extrinsics, optimize_params
from Projection import reprojection_error, project_points_onto_image

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
            H_i = compute_homography(obj_points.copy(), corners.reshape(-1,2))
            hs.append(H_i)
            V_i = format_V(H_i)
            vs.append(V_i)
    
    V = np.vstack(vs)
    b = np.linalg.svd(V)[2][-1]
    K, lamda = extract_intrinsics(b)
    extrinsics = [extract_extrinsics(H, K, lamda) for H in hs]
    K, k1, k2 = optimize_params(K, obj_p, img_p, extrinsics)
    error = reprojection_error(K, (k1,k2), obj_p, img_p, extrinsics)

    fig, axes = plt.subplots(4, 3, figsize=(10,5))
    fig.suptitle('Reprojected Corners On Rectified Image')
    for i, img_path, obj_points, ext, ax in zip(range(len(calib_img_paths)), calib_img_paths, obj_p, extrinsics, axes.flatten()):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        uimg = cv2.undistort(img, K, np.array([k1,k2,0.0,0.0,0.0]))
        project_points_onto_image(uimg, K, (0,0), obj_points, ext)
        uimg = cv2.cvtColor(uimg, cv2.COLOR_BGR2RGB)
        uimg= cv2.rotate(uimg, cv2.ROTATE_90_CLOCKWISE)
        ax.imshow(uimg)
        ax.axis('off')
        # cv2.imshow(f"Img_{i}", img)
        # cv2.waitKey(1)
    
    print("Camera Matrix:\n", K)
    print(f"k1: {k1}, k2: {k2}")
    print(f"Reprojection Error: ", error)
    plt.tight_layout()
    plt.show()
    # while True:
    #     cv2.waitKey(10)
            

if __name__ == '__main__':
    wrapper()