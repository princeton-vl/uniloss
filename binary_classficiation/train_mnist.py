from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import data_loader
from binary_ap import binary_ap
from ap_frombins import  ap_frombins
from torchvision import datasets, transforms
from scipy.interpolate import griddata
from scipy.special import expit

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--save', type=str, default='mnist_uniloss',
                    help='save path')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader, valid_loader = data_loader.get_train_valid_loader('data', args.batch_size, False, 0, shuffle = False)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
nsingle = 16

class UniLoss(torch.autograd.Function):
    def forward(self, pred, target):
        self.save_for_backward(pred, target)
        return torch.zeros(1)

    def backward(self, grad_output):
        pred, target = self.saved_tensors
        grad_input = torch.FloatTensor(pred.size()).fill_(0).cuda()

        bs = pred.size(0) #batch_size
        npos = target.sum() #number of gt positives
        dim = npos * (bs - npos) # number of binaries
        n_good = nsingle
        n_near = nsingle
        n_bad = nsingle
        n_points = nsingle * 3 + 1

        if dim == 0:
            return grad_input, None

        cur_bin_before = np.zeros((dim))     
        idx = -1
        for i in range(bs - 1):
            for j in range(i + 1, bs):
                if target[i] != target[j]: 
                    idx += 1
                    cur_bin_before[idx] = pred[i,0] - pred[j,0]
        sig = expit(cur_bin_before)
        cur_bin = sig * 2 - 1 


        #generate points
        bins = np.zeros((n_points, dim))
        idx = -1
        for i in range(bs - 1):
            for j in range(i + 1, bs):
                if target[i] != target[j]: 
                    idx += 1
                    bins[:, idx] = target[i] - target[j]

        for ib in range(1, n_good + 1): #good
            p1 = random.randint(0, dim - 1)
            bins[ib, p1] = - bins[ib, p1]

        cur_bin_sign = np.sign(cur_bin) #nearby
        diff = np.nonzero(np.not_equal(cur_bin_sign, bins[0]))
        diff = diff[0]
        if len(diff) == 0:
            for ib in range(n_good + 1, n_good + n_near + 1):
                p1 = random.randint(0, dim - 1)
                bins[ib, p1] = - bins[ib, p1]
        else:
            for ib in range(n_good + 1, n_good + n_near + 1):
                bins[ib] = cur_bin_sign.copy()
                p1 = random.randint(0, len(diff) - 1)
                bins[ib, diff[p1]] = - bins[ib, diff[p1]]

        for ib in range(n_good + n_near + 1, n_points): #bad
            bins[ib] = np.sign(np.random.rand(dim) - 0.5)


        values = - np.zeros((n_points))
        for i in range(n_points):
            values[i] = - ap_frombins(bins[i], target)  

        grad_cur_bin = np.zeros((dim))     
        dis = np.zeros((n_points))   
        for ib in range(n_points):
            dis[ib] = np.sqrt(np.sum((bins[ib] - cur_bin)**2))

        sum_wi = np.sum(1 / dis) 
        weighted_ap = np.sum(values / dis)
        for j in range(dim):
            grad_cur_bin[j] = - np.sum(values / (dis ** 3) * (cur_bin[j] - bins[:,j])) / sum_wi - weighted_ap / (sum_wi ** 2) * np.sum((cur_bin[j] - bins[:,j]) / (dis ** 3))
        grad_cur_bin_before = 2 * sig * (1 - sig) * grad_cur_bin

        idx = -1
        for i in range(bs - 1):
            for j in range(i + 1, bs):
                if target[i] != target[j]: 
                    idx += 1
                    grad_input[i] += grad_cur_bin_before[idx]
                    grad_input[j] += -grad_cur_bin_before[idx]

        return grad_input, None

def train(epoch):
    print('Epoch {}'.format(epoch))
    model.train()

    output_all = []
    target_all = []

    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.eq(target, 0).long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        output_all.append(output.data)
        target_all.append(target.data)

        loss = UniLoss()(output, target)
        loss.backward()
        optimizer.step()

    output_all = torch.cat(output_all, 0)
    target_all = torch.cat(target_all, 0)
    ap = binary_ap(output_all, target_all, 0)
    print('training ap = ', ap.item())



def valid():
    model.eval()

    output_all = []
    target_all = []

    for data, target in valid_loader:
        target = torch.eq(target, 0).long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        output_all.append(output.data)
        target_all.append(target.data)

    output_all = torch.cat(output_all, 0)
    target_all = torch.cat(target_all, 0)
    ap = binary_ap(output_all, target_all, 0)
    print('valid ap = ', ap.item())

def test():
    model.eval()

    output_all = []
    target_all = []

    for data, target in test_loader:
        target = torch.eq(target, 0).long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        output_all.append(output.data)
        target_all.append(target.data)


    output_all = torch.cat(output_all, 0)
    target_all = torch.cat(target_all, 0)
    ap = binary_ap(output_all, target_all, 0)
    print('test ap = ', ap.item())


for epoch in range(args.epochs):
    train(epoch)
    valid()
    if (epoch + 1) % 10 == 0: 
        torch.save(model.state_dict(), 'model/' + args.save + '_latest.pth')

test()
