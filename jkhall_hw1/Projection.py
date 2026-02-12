import numpy as np
import cv2

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

def reprojection_error(K, dist, obj_points, img_points, extrinsics):
    error = 0
    for obj_p, img_p, ext in zip(obj_points, img_points, extrinsics):
        pred_p = projection(K, dist, obj_p, ext)
        img_error = np.linalg.norm(img_p-pred_p[:,:2], axis = 1)
        error += np.sum(img_error)
    return error/(len(extrinsics)*obj_p.shape[0])

def project_points_onto_image(img, K, dist, obj_points, extrinsics):
    img_points = projection(K, dist, obj_points, extrinsics)
    for point in img_points:
        u, v = round(point[0]), round(point[1])
        cv2.circle(img, (u, v), 15, (0, 0, 255), 5)