from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import resnet
import data_loader
from torchvision import datasets, transforms
from scipy.special import expit

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--save', type=str, default='cifar10_uniloss',
                    help='save path')
parser.add_argument('--load', type=str, default='none',
                    help='load path')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader, valid_loader = data_loader.get_train_valid_loader('data', 'CIFAR10', args.batch_size, True, 0, shuffle = False)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = resnet.resnet20_cifar(num_classes=10, sig=True)
if args.cuda:
    model.cuda()

if args.load != 'none':
    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = 1e-4)
nsingle = 16



class UniLoss(torch.autograd.Function):
    def forward(self, pred, target):
        self.save_for_backward(pred, target)
        return torch.zeros(1)

    def backward(self, grad_output):
        pred, target = self.saved_tensors
        grad_input = torch.FloatTensor(pred.size()).fill_(0).cuda()

        bs = pred.size(0) #batch_size
        c = pred.size(1)
        dim = bs * (c - 1)
        n_good = nsingle
        n_near = nsingle
        n_bad = nsingle
        n_points = nsingle * 3 + 1

        if dim == 0:
            return grad_input, None

        cur_bin_before = np.zeros((bs, c - 1))   
        for i in range(bs):
            target_now = target[i]
            temp = (pred[i, target_now] - pred[i]).cpu().numpy()
            if target_now > 0: 
                cur_bin_before[i, :target_now] = temp[:target_now]
            if target_now < c - 1:
                cur_bin_before[i, target_now:] = temp[target_now+1:]
        cur_bin_before = cur_bin_before.reshape((dim))
        sig = expit(cur_bin_before)
        cur_bin = sig * 2 - 1 

        #generate points
        bins = np.ones((n_points, dim))

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
        for ib in range(n_points):
            for j in range(bs):
                if bins[ib, j * (c-1) : (j + 1) * (c-1)].sum() == c - 1:
                    values[ib] += - 1
            values[ib] = values[ib] / bs


        grad_cur_bin = np.zeros((dim))     
        dis = np.zeros((n_points))   
        for ib in range(n_points):
            dis[ib] = np.sqrt(np.sum((bins[ib] - cur_bin)**2))
        sum_wi = np.sum(1 / dis) 
        weighted_ap = np.sum(values / dis)
        for j in range(dim):
            grad_cur_bin[j] = - np.sum(values / (dis ** 3) * (cur_bin[j] - bins[:,j])) / sum_wi - weighted_ap / (sum_wi ** 2) * np.sum((cur_bin[j] - bins[:,j]) / (dis ** 3))
        grad_cur_bin_before = 2 * sig * (1 - sig)  * grad_cur_bin

        for ib in range(dim):
            i = ib // (c-1)
            cl = ib - i * (c-1)

            grad_input[i, target[i]] += grad_cur_bin_before[ib]
            if cl  < target[i]:
                grad_input[i, cl] += -grad_cur_bin_before[ib]
            else:
                grad_input[i, cl + 1] += -grad_cur_bin_before[ib]

        return grad_input, None

def train(epoch):
    print('Epoch {}'.format(epoch))
    model.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = UniLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('train acc = ', correct.item() / 45000)

def valid():
    model.eval()

    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('valid acc = ', correct.item() / 5000)

def test():
    model.eval()

    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('test acc = ', correct.item() / len(test_loader.dataset))

def adjust_learning_rate(optimizer, epoch):
    if epoch < 140:
        lr = args.lr
    elif epoch < 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    valid()
    if (epoch + 1) % 50 == 0: 
        torch.save(model.state_dict(), 'model/' + args.save + '_latest.pth')

test()
