import numpy as np
import json
import scipy.io as sio
import torch

with open('mpii_annotations.json') as f:
    data = json.load(f)
anno = sio.loadmat('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
annolist = anno['RELEASE'][0][0][0]
headsize = np.ndarray(shape=(25204,4))
for i in range(len(data)):
    i1 = int(data[i]['annolist_index']-1)
    i2 = int(data[i]['people_index']-1)
    print (i,i1,i2)
    x1 = annolist[0,i1][1][0][i2][0]
    y1 = annolist[0,i1][1][0][i2][1]
    x2 = annolist[0,i1][1][0][i2][2]
    y2 = annolist[0,i1][1][0][i2][3]
    headsize[i] = (x1,y1,x2,y2)

torch.save(headsize, 'headsize.bin')
