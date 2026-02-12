import numpy as np
from scipy.optimize import least_squares

def normalize_pts(pts):
    mean_xy = np.mean(pts, axis=0)
    pts_mean_0 = pts - mean_xy
    distances = np.linalg.norm(pts_mean_0, axis=1)
    avg_dist = np.mean(distances) / np.sqrt(2)
    norm_pts = pts_mean_0 / avg_dist

    norm_pts = np.hstack((norm_pts, np.ones((norm_pts.shape[0], 1))))

    T = np.eye(3)

    T[:2, 2] = -mean_xy

    T = T/avg_dist

    T[2,2] *= avg_dist

    return norm_pts, T

def compute_homography(obj_points, img_points):

    norm_obj, t1 = normalize_pts(obj_points[:,:2])
    norm_img, t2 = normalize_pts(img_points)


    A = np.zeros((2*obj_points.shape[0], 9))

    for i, pt1, pt2 in zip(range(obj_points.shape[0]), norm_obj, norm_img):

        m = np.zeros((2, 9))
        w = pt2[2]*pt1
        y = pt2[1]*pt1
        x = pt2[0]*pt1

        m[0, 3:6] = -w
        m[0, 6:] = y
        m[1, :3] = w
        m[1, 6:] = -x

        A[2*i:2*(i+1), :] = m

    v = np.linalg.svd(A)[2]

    h = v[-1]

    Hp = np.reshape(h, (3,3))
    H_init = np.linalg.inv(t2)@Hp@t1
    H_init/= H_init[2,2]

    H = refine_homography(H_init, obj_points, img_points)
    return H

def reprojection_residuals(h, obj_pts, img_pts):
    H = np.append(h, 1).reshape(3, 3)

    proj = (H @ obj_pts.T).T
    proj = proj[:, :2] / proj[:, [2]]

    return (proj - img_pts).ravel()

def refine_homography(H_init, obj_points, img_points):
    h0 = H_init.flatten()[:-1]
    obj_points[:,2] = 1.0

    result = least_squares(reprojection_residuals, h0, args=(obj_points, img_points), method='lm')

    H_opt = np.append(result.x, 1).reshape(3,3)
    return H_opt

def format_V(H):
    v12 = form_v(H, 0, 1)
    v11 = form_v(H, 0, 0)
    v22 = form_v(H, 1, 1)
    V = np.array([v12, v11-v22])
    return V

def form_v(H, i, j):
    # v = np.array([H[i,0]*H[j,0],
    #               H[i,0]*H[j,1]+H[i,1]*H[j,0],
    #               H[i,1]*H[j,1],
    #               H[i,2]*H[j,0]+H[i,0]*H[j,2],
    #               H[i,2]*H[j,1]+H[i,1]*H[j,2],
    #               H[i,2]*H[j,2]])
    
    v = np.array([H[0,i]*H[0,j],
                  H[0,i]*H[1,j]+H[1,i]*H[0,j],
                  H[1,i]*H[1,j],
                  H[2,i]*H[0,j]+H[0,i]*H[2,j],
                  H[2,i]*H[1,j]+H[1,i]*H[2,j],
                  H[2,i]*H[2,j]])
    return v