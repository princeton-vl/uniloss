from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
from binary_ap import binary_ap
from torchvision import datasets, transforms

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--save', type=str, default='mnist',
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
        self.fc3 = nn.Linear(300, 2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

celoss = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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

        loss = celoss(output, target)
        loss.backward()
        optimizer.step()

    output_all = torch.cat(output_all, 0)
    target_all = torch.cat(target_all, 0)
    ap = binary_ap(output_all, target_all, 1)
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
    ap = binary_ap(output_all, target_all, 1)
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
    ap = binary_ap(output_all, target_all, 1)
    print('test ap = ', ap.item())


for epoch in range(args.epochs):
    train(epoch)
    valid()
    if (epoch + 1) % 10 == 0: 
        torch.save(model.state_dict(), 'model/' + args.save + '_latest.pth')

test()
