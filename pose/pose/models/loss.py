import os, sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pose.utils.evaluation import *

class BinActive(torch.autograd.Function):
    #@staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input
    #@staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class UniLoss(nn.Module):
    def __init__(self, valid=False, a_points=8):
        super(UniLoss, self).__init__()
        self.sig = nn.Sigmoid()
        self.valid = valid
        self.a_points = a_points

    def Nearby_sampling(self, cur_bin, pos_num, neg_num):
        npos = cur_bin.size(1)
        nneg = cur_bin.size(2)
        nparts = cur_bin.size(0)
        points = cur_bin.float().unsqueeze(0).repeat(pos_num+neg_num+1, 1,1,1).view(1+pos_num+neg_num, nparts, -1)
        r1 = torch.randint(npos, [pos_num, nparts])*nneg
        r2 = torch.randint(nneg, [neg_num, nparts])
        r1 = r1.unsqueeze(2).repeat(1,1,nneg) + torch.range(0, nneg-1).view(1,1,-1).repeat(pos_num, nparts, 1)
        r2 = r2.unsqueeze(2).repeat(1,1,npos) + torch.range(0, npos-1).view(1,1,-1).repeat(neg_num, nparts, 1)*nneg
        points[:pos_num] = points[:pos_num].scatter(2, r1.long().cuda(), 1)
        points[pos_num:-1] = points[pos_num:-1].scatter(2, r2.long().cuda(), -1)
        points = points.view(1+pos_num+neg_num, nparts, npos, nneg)
        pck = points.min(3)[0].max(2)[0]/2+0.5
        return points, pck.mean(1)

    def forward(self, pred, meta):
        loss = []
        acc = []
        n_ones = []
        imgsize = pred.size(2)
        bs = pred.size(0)
        cur_bins = []
        pointslist = []
        pcklist = []
        dislist = []
        for i in range(bs):
            bi_target = meta['bi_target'][i].cuda()
            points = meta['points'][i]
            pck = meta['pck'][i]
            tpts = meta['tpts'][i].cuda()
            _pred = pred[i]
            if points is not None:
                points = points.cuda()
            if pck is not None:
                pck = pck.cuda()
            c_idx = (tpts[:,0]>1)*(tpts[:,1]>1)
            npos = bi_target.sum(1).sum(1)
            c_idx *= npos==npos.max()
            nparts = c_idx.sum().item()
            bi_target = bi_target[c_idx]
            npos = npos.max().item()
            nneg = imgsize*imgsize - npos
            _pred = _pred[c_idx].view(nparts, -1)
            if npos == 0:
                npos = 1
                cur_bin = self.sig(0.5 - _pred).view(nparts, npos, nneg)
            else:
                _idx = torch.range(0,imgsize*imgsize-1).cuda().long().unsqueeze(0).repeat(nparts, 1).view(-1)
                pos_idx = _idx[bi_target.view(-1)].view(nparts, -1)
                neg_idx = _idx[1-bi_target.view(-1)].view(nparts, -1)
                pos_val = _pred.gather(1, pos_idx).unsqueeze(2).repeat(1,1,nneg)
                neg_val = _pred.gather(1, neg_idx).unsqueeze(2).repeat(1,1,npos).transpose(2,1)
                if neg_val.is_contiguous() == False:
                    neg_val = neg_val.contiguous()
                cur_bin = self.sig(pos_val - neg_val)
            bi_bin = cur_bin>0.5
            acc.append( (bi_bin).min(2)[0].max(1)[0].sum().float()/nparts )
            cur_bin = cur_bin * 2 - 1
            with torch.no_grad():
                sum_loss = cur_bin.mean()
            a_points, a_pck = self.Nearby_sampling( bi_bin.int()*2-1 , self.a_points, self.a_points)
            points = torch.cat( (points, a_points), 0 )
            pck = torch.cat( (pck, a_pck), 0 )
            _idx = torch.randperm(points.size(0)).cuda()
            cur_bin = cur_bin.view(1,-1).repeat(points.shape[0], 1)
            dis = cur_bin - points[_idx].view(points.size(0), -1)
            dislist.append((dis*dis).sum(1))
            pcklist.append(pck[_idx])
            torch.cuda.empty_cache()
        if not self.valid:
            pck = sum(pcklist)/len(pcklist)
            dis = sum(dislist).sqrt()+1e-8
            wei = 1/dis
            loss = [(wei*pck).sum() / wei.sum() * bs]
        if self.valid:
            return None, sum(acc)/pred.size(0), sum_loss.item()
        else:
            return -sum(loss)/bs, sum(acc)/bs, sum_loss.item()
