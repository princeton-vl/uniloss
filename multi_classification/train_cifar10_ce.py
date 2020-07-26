from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import resnet
import data_loader
from torchvision import datasets, transforms

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--save', type=str, default='cifar10',
                    help='save path')
parser.add_argument('--load', type=str, default='none',
                    help='load path')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='SGD weight decay')
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

model = resnet.resnet20_cifar(num_classes=10, sig=False)
if args.cuda:
    model.cuda()

if args.load != 'none':
    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict)

celoss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.wd)

def train(epoch):
    print('Epoch {}'.format(epoch))
    model.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = celoss(output, target)
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
