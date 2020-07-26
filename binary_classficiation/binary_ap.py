import torch
def binary_ap(pred, target,col):
    target = target.float()
    bs = pred.size(0)
    pred = pred[:,col]

    pred, inds = torch.sort(pred, 0, True)
    npos = target.sum()
    if npos == 0: return 1

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
