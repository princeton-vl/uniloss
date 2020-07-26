from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *

SC_BIAS = 0.6
def mycollate(batch):
    inp = torch.stack([item[0] for item in batch], 0)
    target = torch.stack([item[1] for item in batch] , 0)
    meta = {}
    meta['index'] = torch.LongTensor([item[2]['index'] for item in batch])
    meta['center'] = torch.stack([item[2]['center'] for item in batch], 0)
    meta['scale'] = torch.Tensor([item[2]['scale'] for item in batch])
    meta['pts'] = torch.stack([item[2]['pts'] for item in batch], 0)
    meta['tpts'] = torch.stack([item[2]['tpts'] for item in batch], 0)
    meta['headsize'] = torch.stack([item[2]['headsize'] for item in batch], 0)
    meta['bi_target'] = torch.stack([item[2]['bi_target'] for item in batch], 0)
    meta['points'] = [item[2]['points'] for item in batch]
    meta['pck'] = [item[2]['pck'] for item in batch]
    return inp, target, meta

class Mpii(data.Dataset):
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', thr=0.5, 
                 n_points=32, _idx=[9], direct=False, scale=3):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.scale = scale
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.thr = thr
        self.n_points = n_points
        self._idx = torch.LongTensor(_idx)
        self.direct = direct

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)
        self.headsize = torch.load('data/mpii/headsize.bin')
        self.headsize = torch.from_numpy(self.headsize)
        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        self.mean, self.std = self._compute_mean()
        # load headsize
    def _compute_gt_heatmap(self, h, tpts):
        # compute binary gt headmap
        num_c = tpts.size(0)
        r = (h[2:]-h[:2]).norm()*SC_BIAS
        target = torch.zeros(num_c, self.out_res, self.out_res)
        dis_x = torch.range(0, self.out_res-1).view(1,-1).repeat(num_c, 1)
        dis_y = dis_x.clone()
        dis_x = dis_x - tpts[:,0].view(-1,1).expand_as(dis_x)
        dis_y = dis_y - tpts[:,1].view(-1,1).expand_as(dis_y)
        target = (dis_x*dis_x).unsqueeze(2).repeat(1,1,self.out_res) + (dis_y*dis_y).unsqueeze(2).repeat(1,1,self.out_res).transpose(2,1)
        target = (target.sqrt().double() / r ) < self.thr
        return target.transpose(2,1)
        
    def _generate_bi_points(self, bi_target, tpts):
        c_idx = (tpts[:,0]>1)*(tpts[:,1]>1)
        npos = bi_target.sum(1).sum(1)
        c_idx *= npos==npos.max()
        nparts = c_idx.sum()
        bi_target = bi_target[c_idx]
        npos = npos.max().item()
        nneg = self.out_res*self.out_res - npos
        if npos == 0:
            npos = 1
        flipped_bits = int(npos*3.6-28)
        if flipped_bits < npos*2:
            flipped_bits = int(npos*2)
        _i = int(self.n_points*0.5)
        points = torch.ones(self.n_points+1, nparts, npos*nneg)
        rand_idx_d2 = torch.randint(npos, [_i, nparts, flipped_bits]) 
        rand_idx_d3 = torch.randint(nneg, [_i, nparts, flipped_bits])
        _idx = rand_idx_d2*nneg + rand_idx_d3
        points[:_i] = points[:_i].scatter(2, _idx.long(), -1)
        points[_i:-1] = (torch.rand(self.n_points-_i, nparts, npos*nneg)>0.5).float()*2-1
        points = points.view(self.n_points+1, nparts, npos, nneg)
        pck = points.min(3)[0].max(2)[0]/2+0.5
        return points, pck.mean(1)

    def _compute_mean(self):
        meanstd_file = './data/mpii/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
            h = self.headsize[self.train[index]].clone()
        else:
            a = self.anno[self.valid[index]]
            h = self.headsize[self.valid[index]].clone()
        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']
        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='mpii')
                h[0] = img.size(2) - h[0]
                h[2] = img.size(2) - h[2]
                c[0] = img.size(2) - c[0]
            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)
        pts = pts[self._idx] # pick some parts
        nparts = pts.size(0)
        tpts = pts.clone()
        # Generate ground truth
        target = torch.zeros(nparts, self.out_res, self.out_res)
        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i] = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type, scale=self.scale)
        h[:2] = to_torch(transform(h[:2]+1, c, s, [self.out_res, self.out_res], rot=r) )
        h[2:] = to_torch(transform(h[2:]+1, c, s, [self.out_res, self.out_res], rot=r) )
        if self.direct:
            bi_target = self._compute_gt_heatmap(h, tpts-1)
        else:
            bi_target = torch.zeros(num_c, self.out_res, self.out_res)
        # binary target
        if self.direct:
            c_idx = (tpts[:,0]>1)*(tpts[:,1]>1)
            if c_idx.sum().item() == 0:
                return self.__getitem__(random.randint(0, self.__len__()-1))   
            points, pck = self._generate_bi_points(bi_target, tpts)
        else:
            points = None
            pck = None
        # Meta info
        meta = {'points' : points, 'pck' : pck, 'index' : index, 'center' : c, 'scale' : s, 'pts' : pts, 'tpts' : tpts, 'headsize' : h, 'bi_target' : bi_target}
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images'),
        batch_size=10, shuffle=False, num_workers=10, collate_fn=mycollate)
    for i, (inputs, target, meta) in enumerate(train_loader):
        if i % 100 == 0:
            print (i)
