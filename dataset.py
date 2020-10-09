import torch
import os
import numpy as np
from torch.utils import data

class VituralGraphTransform(object):
    def __init__(self, data):
        pass

class VituralGraphDataset(data.Dataset):
    def __init__(self, root, pic_size='640x480', points_num=64, target_transform=None):
        self.root = os.path.join(root, pic_size, str(points_num))
        self.graph_keys = ['keypoints', 'scores', 'descriptors']
        self.out_keys = ['matches', 'matching_scores']
        self.target_transform = target_transform
        self.graphlist = list(sorted(os.listdir(self.root)))

    def __getitem__(self, index):
        graph_data = np.load(os.path.join(self.root,self.graphlist[index]), allow_pickle=True).item()
        
        if self.target_transform is not None:
            pass
        
        # graph_info = torch.cat((torch.cat((raw_data[i+'0'] for i in self.graph_keys), 1),
        #                         torch.cat((raw_data[i+'1'] for i in self.graph_keys), 1)), 1)
        # target = torch.cat((torch.cat((raw_data[i+'0'] for i in self.out_keys),1),
        #                     torch.cat((raw_data[i+'0'] for i in self.out_keys),1)), 1)
        # base_pose = raw_data['base_pose']
        target = dict(**{i+'0': graph_data[i+'0'] for i in self.out_keys},
                      **{i+'1': graph_data[i+'1'] for i in self.out_keys})
        base_pose = graph_data['base_pose']
        return graph_data, target, base_pose

    def __len__(self):
        return len(self.graphlist)

