from sklearn.linear_model import RANSACRegressor
import numpy as np
from scipy.spatial.transform import Rotation as R

def eliminate_ground(X):
    x = X[:,:2]
    z = X[:,2]
    reg = RANSACRegressor().fit(x, z)
    groud_mask = reg.inlier_mask_
    return groud_mask

def rotate_y(degree):
    rad = degree * np.pi / 180
    return R.from_rotvec(rad * np.array([0, 1, 0])).as_matrix()
def rotate_z(degree):
    rad = degree * np.pi / 180
    return R.from_rotvec(rad * np.array([0, 0, 1])).as_matrix()
def rotate_x(degree):
    rad = degree * np.pi / 180
    return R.from_rotvec(rad * np.array([1, 0, 0])).as_matrix()