from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import pose.utils.log as log

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from pose import Bar
from pose.utils.logger import print_args, Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0

def main(args):
    global best_acc
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    _logger = log.get_logger(__name__, args)
    _logger.info(print_args(args))
    
    # create model
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=len(args.index_classes))

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = models.loss.UniLoss(a_points=args.a_points)
    optimizer = torch.optim.RMSprop(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'mpii-' + args.arch
    if args.resume:
        if isfile(args.resume):
            _logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                print (param_group['lr'])
            _logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=False)
            logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])
        else:
            _logger.info("=> no checkpoint found at '{}'".format(args.resume))
    else:        
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    cudnn.benchmark = True
    _logger.info('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images', 
                      sigma=args.sigma, label_type=args.label_type, _idx=args.index_classes, direct=True, n_points=args.n_points),
        batch_size=args.train_batch, shuffle=True, collate_fn=datasets.mpii.mycollate,
        num_workers=args.workers, pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                      sigma=args.sigma, label_type=args.label_type, _idx=args.index_classes, train=False, direct=True),
        batch_size=args.test_batch, shuffle=False, collate_fn=datasets.mpii.mycollate,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        _logger.warning('\nEvaluation only') 
        loss, acc, predictions = validate(val_loader, model, criterion, len(args.index_classes), False, args.flip, _logger, evaluate_only=True)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    # multi-thread
    inqueues = []
    outqueues = []
    valid_accs = []
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        _logger.warning('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *=  args.sigma_decay
            val_loader.dataset.sigma *=  args.sigma_decay
        # train for one epoch
        train_loss, train_acc = train(inqueues, outqueues, train_loader, model, criterion, optimizer, args.debug, args.flip, args.clip, _logger)
        # evaluate on validation set
        with torch.no_grad():
            valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, len(args.index_classes),
                                                      args.debug, args.flip, _logger)
        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])
        valid_accs.append(valid_acc)   
        if args.schedule[0] == -1:
            if len(valid_accs) > 8:
                if sum(valid_accs[-4:])/4*0.99 < sum(valid_accs[-8:-4])/4:
                    args.schedule.append(epoch+1)
                    valid_accs = []
        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint, snapshot=1)

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))

def train(inqueues, outqueues, train_loader, model, criterion, optimizer, debug=False, flip=True, clip=1, _logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    sum_losses = AverageMeter()
    # switch to train mode
    model.train()
    criterion.valid = False
    end = time.time()
    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(train_loader))
    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if True:
            input_var = torch.autograd.Variable(inputs.cuda())
            target_var = torch.autograd.Variable(target.cuda(async=True))
            # compute output
            output = model(input_var)
            if len(inqueues) > 0: 
                loss = []
                acc = []
                sum_loss = []
                for j in range(len(output)):
                    grad = []
                    for ii in range(output[0].size(0)):
                        data = {}
                        data['output'] = output[j][ii]
                        data['meta'] = {}
                        data['meta']['bi_target'] = meta['bi_target'][ii]
                        data['meta']['pck'] = meta['pck'][ii]
                        data['meta']['points'] = meta['points'][ii]
                        data['meta']['tpts'] = meta['tpts'][ii]
                        inqueues[ii].send(data)
                    for ii in range(output[0].size(0)):
                        _loss, _acc, _sum_loss, _grad = outqueues[ii].recv()
                        loss.append(_loss)
                        grad.append(_grad)
                        if j == len(output)-1:
                            acc.append(_acc)
                            sum_loss.append(_sum_loss)
                    optimizer.zero_grad()
                    output[0].backward(torch.stack(grad, 0))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                loss = sum(loss)/output[0].size(0)
                acc = sum(acc)/output[0].size(0)
                sum_loss = sum(sum_loss)/output[0].size(0)
            else:
                optimizer.zero_grad()
                loss, acc, sum_loss = criterion(output[0], meta)
                for j in range(1, len(output)):
                    _loss, acc, sum_loss = criterion(output[j], meta)
                    loss += _loss
                loss.backward()
                optimizer.step()

            if debug: # visualize groundtruth and predictions
                gt_batch_img = batch_with_heatmap(inputs, target)
                pred_batch_img = batch_with_heatmap(inputs, score_map)
                if not gt_win or not pred_win:
                    ax1 = plt.subplot(121)
                    ax1.title.set_text('Groundtruth')
                    gt_win = plt.imshow(gt_batch_img)
                    ax2 = plt.subplot(122)
                    ax2.title.set_text('Prediction')
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(loss.data.item(), inputs.size(0))
            acces.update(acc.item(), inputs.size(0))
            sum_losses.update(sum_loss)
        loss = None
        torch.cuda.empty_cache()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | Sum_Loss: {sum_loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    loss=losses.avg*100,
                    sum_loss=sum_losses.avg*100,
                    acc=acces.avg*100
                    )
        _logger.info(bar.suffix)
    bar.finish()
    return losses.avg*100, acces.avg*100


def validate(val_loader, model, criterion, num_classes, debug=False, flip=True, _logger=None, evaluate_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acces = AverageMeter()
    sum_losses = AverageMeter()
    criterion.valid = True
    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)
    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(inputs.cuda() )

        # compute output
        output = model(input_var)
        score_map = output[-1].data.cpu()
        if flip:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(), 
                    volatile=True
                )
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output
        if not evaluate_only:
            _, acc, sum_loss = criterion(output[-1], meta)
            acces.update(acc.item(), inputs.size(0))
            sum_losses.update(sum_loss)
        # generate predictions
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Sum_Loss: {sum_loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    sum_loss=sum_losses.avg*100,
                    acc=acces.avg*100
                    ))

        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | Sum_Loss: {sum_loss:.4f} | Acc: {acc: .4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                sum_loss=sum_losses.avg*100,
                acc=acces.avg*100
                )
        _logger.info(bar.suffix)
    bar.finish()
    return sum_losses.avg*100, acces.avg*100, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--index-classes', type=int, nargs='+', default=[9],
                        help='Indices of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 40, 50],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.25,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--log', required=True, help='path to log file')
    parser.add_argument('--n-points', type=int, default=32, help='Number of points to sample')
    parser.add_argument('--a-points', type=int, default=8, help='Number of adaptive points')
    parser.add_argument('--clip', type=float, default=5, help='Clip gradients')
    main(parser.parse_args())
