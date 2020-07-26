import torch
import numpy as np
def ap_frombins(bins, target):
    target = target.float()
    bs = target.size(0)
   
    linked = [[] for i in range(bs)]
    above = np.zeros((bs))
    idx = -1
    for i in range(bs - 1):
        for j in range(i + 1, bs):
            if target[i] != target[j]: 
                idx += 1
                if bins[idx] > 0: # score_i is larger than score_j 
                    linked[i].append(j)
                    above[j] += 1
                else:
                    linked[j].append(i)
                    above[i] += 1

    inds = []
    while np.max(above) >= 0:
        if not 0 in above: 
            break
        for i in range(bs):
            if above[i] == 0:
                inds.append(i)
                for j in linked[i]:
                    above[j] -= 1                        
                above[i] = -1
                break

    if np.max(above) >= 0: 
        ##invalid binaries
        return 0
    npos = target.sum()
    if npos == 0: 
        return 1

    rec = torch.zeros(bs + 2)
    prec = torch.zeros(bs + 2)
    rec[0] = 0
    rec[bs + 1] = 1
    prec[1] = 0
    prec[bs + 1] = 0
    sumtp = 0
    sumfp = 0
    for i in range(bs):
        tp = target[inds[i]]
        fp = 1 - tp
        sumtp = sumtp + tp
        sumfp = sumfp + fp
        rec[i + 1] = sumtp / npos
        prec[i + 1] = sumtp / (sumtp + sumfp)

    for i in range(bs, -1 , -1):
        prec[i] = max(prec[i], prec[i + 1]) 
    ap = 0
    for i in range(bs):
        ap = ap + (rec[i+1] - rec[i]) * prec[i]
    return ap

