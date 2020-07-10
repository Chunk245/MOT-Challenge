import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.join(root_dir, 'src'))
from tracker.data_track import MOT16Sequences
from tracker.data_obj_detect import MOT16ObjDetect
from tracker.object_detector import FRCNN_FPN
from tracker.tracker import Tracker
from tracker.utils import (plot_sequence, evaluate_mot_accums, get_mot_accum,
                           evaluate_obj_detect, obj_detect_transforms, build_crops)
import random 
import torchvision.models as models
import torch.optim as optim

import motmetrics as mm
mm.lap.default_solver = 'lap'

root_dir = "/home/teamcv/Downloads/xuexi/MOT/"
seed = 12345
seq_name = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11']
data_dir = os.path.join(root_dir, 'data/MOT16')
output_dir = os.path.join(root_dir, 'output')

sequences = []
for seq_n in seq_name:
    sequences.append(MOT16Sequences(seq_n, data_dir))

class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        x = self.resnet_layer(x)
        return x
        
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean(), distance_positive.mean(), distance_negative.mean()
        
vis_threshold = 0.25
ped_roi = {}

for sequence in sequences:
    for seq in sequence:
        print("Preparing sequence: {}".format(seq))

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)

        ped_roi[seq] = {}

        for frame in tqdm(data_loader):
            if 'gt' in frame.keys():
                gt = frame['gt']
                resized_rois = build_crops(frame['img'], gt.values())
                for idx, (gt_id, box) in enumerate(gt.items()):
                    if frame['vis'][gt_id] > vis_threshold:
                        if gt_id not in ped_roi[seq].keys():
                            ped_roi[seq][gt_id] = []
                        if not resized_rois[idx][0][0][0] == -1:
                            ped_roi[seq][gt_id].append(resized_rois[idx].unsqueeze(0))

epochs = 10

model = Net(models.resnext50_32x4d(pretrained=True))
model.to('cuda')
criterion = TripletLoss(margin = 0.2)

optimizer = optim.Adam(model.parameters(), lr=0.001)
for ep in range(epochs):
    for seq in ped_roi.keys():        
        loss_ep = 0
        dis_p_ep = 0
        dis_n_ep = 0
        ped_roi_ss = ped_roi[seq]      
        ped_roi_s = {}
        for tmp in ped_roi_ss.keys():
            if len(ped_roi_ss[tmp])>0:
                ped_roi_s[tmp] = ped_roi_ss[tmp]
        
        gt_ids = list(ped_roi_s.keys())
        
        for anp_idx in ped_roi_s.keys():    
            anp = ped_roi_s[anp_idx]    

            
            t = gt_ids.copy()
            t.remove(anp_idx)                
            neg = ped_roi_s[t[random.randint(0,len(t)-1)]]              
            iter_pperson = min(len(neg), len(anp))
            iter_pperson = min(iter_pperson, 20)
            
            idxs = np.arange(len(anp))
            np.random.shuffle(idxs)
            aa = []
            for i in idxs[:iter_pperson]:
                aa.append(anp[i].squeeze(0))

            pp = []
            for i in idxs[-iter_pperson:]:
                pp.append(anp[i].squeeze(0))                

            idxs = np.arange(len(neg))
            np.random.shuffle(idxs)
            nnn = []
            for i in idxs[:iter_pperson]:
                nnn.append(neg[i].squeeze(0))
            
            inimg =aa +pp+nnn
            imgs = torch.stack(inimg).cuda()
            optimizer.zero_grad()
            out = model(imgs)
            sp = iter_pperson
            
            loss_it, dis_p_it, dis_n_it = criterion(out[:sp], out[sp:sp*2], out[sp*2:sp*3])   
            loss_it.backward()
            optimizer.step()     
            
            loss_ep += loss_it
            dis_p_ep += dis_p_it
            dis_n_ep += dis_n_it
        loss_ep /= len(ped_roi_s.keys())
        dis_p_ep /= len(ped_roi_s.keys())
        dis_n_ep /= len(ped_roi_s.keys())
        print('epoch [{}/{}] seq {}: loss:{:.5f} distance_pos:{:.5f} distance_neg:{:.5f}'.format  
              (ep+1, epochs, seq, loss_ep, dis_p_ep, dis_n_ep))  
    torch.save(model, 'ep_{}.pth'.format(ep))
