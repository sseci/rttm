import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import csv
import json
from models import DummyModel
from utils import eliminate_ground, rotate_y, rotate_z, rotate_x

# get point cloud
def get_pc(idx, crop=False, floor=None, close=None, far=None, ceil=None, left=None, right=None, path=None):
    if path is None:
        pc = np.fromfile("%s/%06d.bin" % ("dataset_3/ordered", idx), dtype=np.float32).reshape(-1, 4)
        #  pc = np.fromfile("pcd.bin", dtype=np.float32).reshape(-1, 4)
    else:
        pc = np.fromfile("%s/%06d.bin" % (path, idx), dtype=np.float32).reshape(-1, 4)
    # all rows, 3 cols

    pc = pc[:, :3]
    ry = np.transpose(rotate_y(1))
    rz = np.transpose(rotate_z(5.5))
    rx = np.transpose(rotate_x(-1))
    temp = np.matmul(pc, ry)
    temp = np.matmul(temp, rz)
    temp = np.matmul(temp, rx)

    if floor is not None:
        temp = temp[np.where(temp[:,2] >= floor)]
    if ceil is not None:
        temp = temp[np.where(temp[:,2] <= ceil)]
    if close is not None:
        temp = temp[np.where(temp[:,1] >= close)]
    if far is not None:
        temp = temp[np.where(temp[:,1] <= far)]
    if right is not None:
        temp = temp[np.where(temp[:,0] <= right)]
    if left is not None:
        temp = temp[np.where(temp[:,0] >= left)]

    if crop:
        temp = temp[:,:2]

    return temp

def get_by_idx(idx):
    return get_pc(idx, crop=True, close=8, far=26, floor=-3.26, left=-50, right=50)

def readlabel(idx):
    reader = csv.reader(open("%s/%06d.txt" % ("dataset2", idx)), delimiter=" ")
    bbx_list = []
    ry = rotate_y(3.3)
    rz = rotate_z(-6.5)
    for row in reader:
        label = row[0]
        xctr = float(row[1])
        yctr = float(row[2])
        zctr = float(row[3])
        xlen = float(row[4])
        ylen = float(row[5])

        center = np.array([[xctr], [yctr], [zctr]])
        center = rz @ ry @ center
        bbx_list.append([center[0] - xlen/2, center[1] - ylen/2, xlen, ylen, label])
    return bbx_list

def visualize(pc, axis=False):
    print(pc.shape)
    if axis:
        x = x_axis()
        y = y_axis(0, 0)
        print(x.shape)
        pc = np.concatenate((pc, x))
        pc = np.concatenate((pc, y))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud("vis.ply", pcd)

    pcd_load = o3d.io.read_point_cloud("vis.ply")
    o3d.visualization.draw_geometries([pcd_load])

def plt_np(pc):
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, s=0.1)
    plt.show()

def x_axis():
    x = np.expand_dims(np.array([i for i in range(50)]), axis=0)
    y = np.expand_dims(np.array([0 for i in range(50)]), axis=0)
    z = np.expand_dims(np.array([0 for i in range(50)]), axis=0)
    x=np.append(x, y, axis=0)
    x=np.append(x, z, axis=0)
    return np.transpose(x)

def y_axis(x, z):
    x = np.expand_dims(np.array([x for i in range(50)]), axis=0)
    y = np.expand_dims(np.array([i for i in range(50)]), axis=0)
    z = np.expand_dims(np.array([z for i in range(50)]), axis=0)
    x=np.append(x, y, axis=0)
    x=np.append(x, z, axis=0)
    return np.transpose(x)

def projection(pc, lane_mid):
    # pc: (n, 3) array
    # return: (k) array, where k <= n
    # If we have 3 lanes, we will run projection on 3 different lane_mid, the sum of each projection()'s length should be n
    return None

def show_result(idx):
    pc = get_pc(idx, close=8, far=26, floor=-1.2, left=-50, right=50)
    pc = pc[:, [0, 1]]
    cluster = DBSCAN(eps=1, min_samples=2).fit(pc)
    labels = cluster.labels_
    
    for i in range(labels.max()+1):
        group = pc[np.where(labels==i), :][0]
        _ = plt.scatter(group[:, 0], group[:, 1], s=0.5)

    bbx_list = readlabel(idx)
    _ = plt.axis('scaled')
    plt.xlim([-60, 60])
    plt.ylim([0, 30])
    ax = plt.gca()
    for bbx in bbx_list:
        rect = patches.Rectangle((bbx[0], bbx[1]), bbx[2], bbx[3], linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    # plt.savefig(f"phase{idx}")
    plt.pause(0.1)
    plt.clf()


class LabelConvert:
    def fit_frame(self, X, clusters, bbx_list):
        # Main function, convert bbx to label of cluster
        # X: xz cloud point, (2, n) arrays
        # clusters: label for cluster (1, n) array
        # bbx_list: a list of bounding box. Example: bbx_list[0] = ["truck", x_left, x_right, z_close, z_far]
        # Return label_list: (c) array, where c is number of clusters. 0: truck, 1: suv, 2: noise

        label_list = np.zeros(clusters.max() + 1)
        for i in range(clusters.max()+1):
            group = X[np.where(clusters==i), :][0]
            label_list[i] = self.assign_label(group, bbx_list)
        return label_list

    def is_in(self, x, bbx):
        if x[0] >= bbx[1] and x[0] <= bbx[2] and x[1] >= bbx[3] and x[1] <= bbx[4]:
            return True
        else :
            return False

    def check_label(self, x, bbx_list, label):
        for bbx in bbx_list:
            label_type = bbx[0]
            if self.is_in(x, bbx):
                label[label_type] += 1
                continue

    def assign_label(self, X, bbx_list):
        label = {
            "truck":0,
            "suv":0,
            "noise":0
        }
        for x in X:
            self.check_label(x, bbx_list, label)

        if label["truck"] >= label["suv"] and label["truck"] >= label["noise"]:
            return 0
        if label["suv"] >= label["truck"] and label["suv"] >= label["noise"]:
            return 1
        return 2

class Config:
    def __init__(self, highway):
        self.highway = highway

class Highway:
    def __init__(self, D_close, Lane_width, Lane_count):
        self.Dc = D_close
        self.lane_count = Lane_count
        self.lanes_left = np.array([D_close + i * Lane_width for i in range(Lane_count)])
        self.lanes_right = self.lanes_left + Lane_width
        self.lanes_mid = self.lanes_left + 0.5 * Lane_width

class Vehicle:
    def __init__(self, label, coord, vid=None, dim=None):
        self.label = label # which type of vehicle
        self.coord = coord # np.array([x, y, z])
        self.dim = dim     # np.array([width, length, height])
        self.vid = vid
        self.lane = None

    def get_coord(self):
        return self.coord

    def get_label(self):
        return self.label

    def has_vid(self):
        return not self.vid is None

    def has_dim(self):
        return not self.dim is None

    def get_vid(self):
        assert self.has_vid(), "vehicle id is None"
        return self.vid

    def add_lane(self, lane):
        self.lane = lane

    def locate_lane(self):
        lane = 0
        dist = 1000
        for i in range(config.highway.lane_count):
            cur_dist = self.coord[0] - config.highway.lanes_mid[i]
            if cur_dist < dist:
                lane = i
                dist = cur_dist
        self.add_lane(lane)
        return lane

    def volumn(self):
        assert self.has_dim(), "Dimmension is None"
        return np.prod(self.dim)

### Start Glabal Parameters ###
Dc = 1
Lw = 3.6576
Lc = 3
config = Config(Highway(Dc, Lw, Lc))
# Global to all frames to save memory 

class FramePrediction:
    def __init__(self, fid, count_lane=False):
        self.fid = fid
        self.count = 0
        self.predictions = []
        self.lane_count = np.zeros(config.highway.lane_count)
        self.count_lane = count_lane

    def add_prediction(self, vehicle):
        if self.count_lane:
            self.lane_count[vehicle.locate_lane()] += 1
        self.predictions.append(vehicle)
        self.count += 1

    def vehicle_count(self):
        return self.count

    def get_lane_count(self):
        return self.lane_count







if __name__ == '__main__':

    # pc = get_pc(2695, close=8, far=26, floor=-1.2, left=-50, right=50)
    # pc = get_pc(2695, close=8, far=26)

    idx = 1
    # pc = get_pc(idx, path="dataset_3/ordered")
    pc = get_pc(idx)
    print(pc.shape)
    groud_mask = eliminate_ground(pc)
    print(groud_mask)
    visualize(pc, True)

    pc2 = get_by_idx(idx)
    print(pc2.shape)
    # groud_mask = eliminate_ground(pc2)
    # print(groud_mask)
    # visualize(pc2, True)
    # visualize(pc[groud_mask], True)

    # for idx in range(1, 1000):
    #     pc = get_pc(idx, path="dataset_3/ordered")
    #     groud_mask = eliminate_ground(pc)
    #     print(groud_mask)
    #     visualize(pc, True)

    idx = 1
    model = DummyModel(9, 30)
    for i in range(idx, idx + 450, 2):
    # for i in range(idx, idx + 1, 2):
        pc = get_by_idx(i)
        pc2 = get_pc(i, crop=True, close=8, far=26, left=-50, right=50)
        print("idx:", i)
        print("pc.shape:", pc.shape)
        print("pc2.shape:", pc2.shape)
        # bbx_list = readlabel(i)
        bbx_list = None
        model.show_result(pc, i, pc2, animate=True, show_lane=True)
        # visualize(pc, True)
        # model.show_result(pc, i, pc2, bbx_list=bbx_list, animate=True, show_lane=True)

    

    '''
    model = DummyModel(9, 30)

    idx = 0
    for i in range(idx, idx + 3000, 2):
        pc = get_by_idx(i)
        pc2 = get_pc(i, crop=True, close=8, far=26, left=-50, right=50)
        bbx_list = readlabel(i)
        bbx_list = []
        model.show_result(pc, i, pc2, bbx_list=bbx_list, animate=True, show_lane=True)
    '''