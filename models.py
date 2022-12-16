import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import json

DUMP = True
LW = 3.65

class DummyModel:
    def __init__(self, length_threshold, noise_threshold):
        self.lt = length_threshold
        self.nt = noise_threshold
        self.translation = {
            0 : "Noise",
            1 : "Vehicle",
            2 : "Truck"
        }

    def predict(self, X):
        if X.shape[0] < self.nt:
            return 0
        if (X[:,0].max() - X[:,0].min()) < self.lt:
            return 1
        return 2

    def translate_label(self, label):
        return self.translation[label]

    def parse_group(self, group):
        x_max = group[:, 0].max() + 1
        x_min = group[:, 0].min() - 1
        y_max = group[:, 1].max() + 1
        y_min = group[:, 1].min() - 1
        xrange = (x_max - x_min)
        yrange = (y_max - y_min)
        return [(x_min, y_min), xrange, yrange]

    def reduce_y(self, X):
        x = X.copy()
        x[:, 1] = x[:, 1] - (10 - 0.5*LW)
        for i in range(x.shape[0]):
            x[i, 1] = x[i, 1] // LW
        return x

    def show_result(self, pc, idx, pc2=None, bbx_list=None, animate=False, show_lane=False):
        kernal = np.array([[0.3, 0], [0, 2]])
        phi = self.reduce_y(pc) @ kernal
        cluster = DBSCAN(eps=1, min_samples=20).fit(phi)
        labels = cluster.labels_
        ax = plt.gca()

        if DUMP:
            result = {
                "fid": idx,
                "preds": []
            }
        if pc2 is None:
            pc2 = pc.copy()
        pc2[:,1]  = pc2[:,1] + 30
        plt.scatter(pc2[:, 0], pc2[:, 1], s=0.5)

        if show_lane:
            for i in range(5):
                lane,  = plt.plot((40, -40), (10-0.5*LW+LW*i, 10-0.5*LW+LW*i), color='grey', linewidth=0.5)
            start, = plt.plot((-30, -30), (10-0.5*LW, 10+3.5*LW), color="red", linewidth=0.5)
        
    
        for i in range(labels.max()+1):
            group = pc[np.where(labels==i), :][0]
            label = self.predict(group)
            _ = plt.scatter(group[:, 0], group[:, 1], s=0.5, color='green')

            center, xr, yr = self.parse_group(group)
            if label > 0 and (center[0] + xr >= -30):
                rect = patches.Rectangle(center, xr, yr, linewidth=1,edgecolor='r',facecolor='none')
                pred = ax.add_patch(rect)
                text_label = self.translate_label(label)
                plt.text(center[0], center[1], text_label, fontsize=6)
                
                if DUMP:
                    veh = {
                        "x": center[0],
                        "y": center[1],
                        "label": text_label
                    }
                    result["preds"].append(veh)

        _ = plt.axis('scaled')
        plt.xlim([-60, 60])
        plt.ylim([0, 80])
        ax = plt.gca()
        if bbx_list is not None:
            for bbx in bbx_list:
                rect = patches.Rectangle((bbx[0], bbx[1]+30), bbx[2], bbx[3], linewidth=0.5,edgecolor='blue',facecolor='none')
                gt = ax.add_patch(rect)
                plt.text(bbx[0], bbx[1]+30, bbx[4], fontsize=6)
        
        # gt_placeholder = patches.Rectangle((-30, 0), 0.1, 0.1, linewidth=0.5,edgecolor='blue',facecolor='none')
        # # pred_placeholder = patches.Rectangle(center, xr, yr, linewidth=1,edgecolor='r',facecolor='none')
        # gt = ax.add_patch(gt_placeholder)
        # pred = ax.add_patch(pred_placeholder)
        # plt.legend([pred, gt, lane, start], ["prediction", "gt_label", "lane", "threshold"], loc="upper left")
        # # plt.savefig(f"phase")
        if animate:
            plt.pause(0.1)
            plt.clf()
        else:
            plt.show()

        if DUMP:
            f = open("result.json", "a")
            f.write(json.dumps(result))
            f.write("\n")
            f.close()
