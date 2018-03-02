import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from itertools import *
import models
from LBA import P_ab, P_aba, equality_matrix
from embedding_to_logit import embedding_to_logit
import torch.nn.functional as F
from EqPortionSampler import EqSampler
from tensorboardX import SummaryWriter

writer = SummaryWriter()
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('target', metavar='DIR2',
                    help='path to target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-s', '--split', default=128, type=int,
                    metavar='N', help='batch split index (default: 128)')
parser.add_argument('-nc', '--num_classes', default=31, type=int,
                    metavar='N', help='class number (default: 31)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')

CUDA_LAUNCH_BLOCKING=1
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.arch = 'alexnet'
    args.distributed = args.world_size > 1
    args.alpha1 = torch.ones(1, 1).cuda(async=True)
    args.alpha1 = torch.autograd.Variable(args.alpha1, requires_grad=True)
    args.alpha2 = torch.ones(1, 1).cuda(async=True)
    args.alpha2 = torch.autograd.Variable(args.alpha2, requires_grad=True)
    args.alpha3 = torch.ones(1, 1).cuda(async=True)
    args.alpha3 = torch.autograd.Variable(args.alpha3, requires_grad=True)
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.alexnet_da(pretrained=True, batch_size = args.batch_size, split = args.split)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.alexnet_da(pretrained=False, batch_size = args.batch_size, split = args.split)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_t = nn.NLLLoss().cuda()

    optimizer = torch.optim.SGD([{'params': model.features.parameters()},
                                 {'params': args.alpha1, 'lr': 1e-3},
                                 {'params': args.alpha2, 'lr': 1e-3},
                                 {'params': args.alpha3, 'lr': 1e-3},
                                 {'params': model.fc6.parameters()},
                                 {'params': model.fc7.parameters()},
                                 {'params': model.fc8.parameters(), 'lr': 1e-2}],
                                lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # Data loading code
    traindir_s = os.path.join(args.data, 'images')
    traindir_t = os.path.join(args.target, 'images')
    valdir = os.path.join(args.target, 'images')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset_s = datasets.ImageFolder(
        traindir_s,
        transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_dataset_t = datasets.ImageFolder(
        traindir_t,
        transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset_s = datasets.ImageFolder(
        traindir_s,
        transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler_s = torch.utils.data.distributed.DistributedSampler(train_dataset_s)
        train_sampler_t = torch.utils.data.distributed.DistributedSampler(train_dataset_t)
    else:
        train_sampler_s = EqSampler(train_dataset_s)
        train_sampler_t = EqSampler(train_dataset_t)

    train_loader_s = torch.utils.data.DataLoader(
        train_dataset_s, batch_size=args.split, shuffle=(train_sampler_s is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_s, drop_last=True)
    train_loader_t = torch.utils.data.DataLoader(
        train_dataset_t, batch_size=args.batch_size-args.split, shuffle=(train_sampler_t is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_t, drop_last=True)

    val_loader_s1 = torch.utils.data.DataLoader(
        val_dataset_s, batch_size=args.split, shuffle=(train_sampler_s is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_s, drop_last=True)
    val_loader_s2 = torch.utils.data.DataLoader(
        val_dataset_s, batch_size=args.split, shuffle=(train_sampler_s is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_s, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=200, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    validate(val_loader, val_loader_s1, val_loader_s2, model, criterion, args.alpha1, args.alpha2, args.alpha3, args.split, 2, -1)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler_s.set_epoch(epoch)
            train_sampler_t.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader_s, train_loader_t, model, criterion, criterion_t, optimizer, epoch, epoch, args.alpha1, args.alpha2, args.alpha3, args.split, args.num_classes)

        # evaluate on validation set
        prec1 = validate(val_loader, val_loader_s1, val_loader_s2, model, criterion, args.alpha1, args.alpha2, args.alpha3, args.split, 2, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        args.alpha1.data = torch.min(torch.max(args.alpha1.data, 0.5 * torch.ones(1).cuda()), torch.ones(1).cuda())
        args.alpha2.data = torch.min(torch.max(args.alpha2.data, 0.5 * torch.ones(1).cuda()), torch.ones(1).cuda())
        args.alpha3.data = torch.min(torch.max(args.alpha3.data, 0.5 * torch.ones(1).cuda()), torch.ones(1).cuda())

    print validate(val_loader, val_loader_s1, val_loader_s2, model, criterion, args.alpha1, args.alpha2, args.alpha3, args.split, 30, epoch+1)



def train(train_loader_s, train_loader_t, model, criterion, criterion_t, optimizer, epoch, num_epochs, alpha1, alpha2, alpha3, split, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    m = nn.LogSoftmax()
    n = nn.Softmax()
    l = nn.Sigmoid()
    k = nn.MSELoss()

    # switch to train mode
    model.train()

    #freeze conv layers
    if epoch <120:
        freeze = 0
        for param in model.parameters():
            if freeze < 10:
                param.requires_grad = False
                freeze += 1

    end = time.time()

    V = torch.div(torch.ones(split, args.batch_size - split), args.batch_size - split)
    V = V.cuda(async=True)
    V = torch.autograd.Variable(V)

    for tuple in izip(enumerate(train_loader_s), enumerate(train_loader_t)):
        i, (input_s, target_s) = tuple[0]
        j, (input_t, target_t) = tuple[1]

        data_time.update(time.time() - end)

        input_var_s = torch.autograd.Variable(input_s)
        input_var_t = torch.autograd.Variable(input_t)
        input_var_s.requires_grad = True
        input_var_t.requires_grad = True
        input = torch.cat((input_var_s, input_var_t),0)

        target_var_s = torch.autograd.Variable(target_s)
        target_var_t = torch.autograd.Variable(target_t)
        target_var_s = target_var_s.cuda()
        target_var_t = target_var_t.cuda()
        target = torch.cat((target_s, target_t), 0)

        eq_mat = equality_matrix(target_s, num_classes).cuda()
        eq_mat = eq_mat.cuda(async=True)
        eq_mat = torch.autograd.Variable(eq_mat)

        # compute output
        output = model(input, alpha1, alpha2, alpha3)

        output_s = output.narrow(0, 0, split)
        output_t = output.narrow(0, split,  args.batch_size -  split)

        output_sm_s = l(output_s)
        output_sm_t = l(output_t)
        p_aba = P_aba(output_sm_s, output_sm_t).cuda(async=True)
        loss_walker = k(p_aba, eq_mat).cuda(async=True)

        loss_s = criterion(m(output_s), target_var_s)
        loss_t = - torch.mean(torch.sum(m(output_t) * n(output_t),1))
        if epoch < 60:
            loss = loss_s + 0.1 * loss_t
        else:
            loss = loss_s + 0.1 * loss_t + 0.1 * loss_walker

        prec1, prec5 = accuracy(output_s.data, target_s.cuda(async=True), topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                     epoch, i, len(train_loader_s), batch_time=batch_time,
                     data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, val_loader_s1, val_loader_s2, model, criterion, alpha1 , alpha2, alpha3, split, epoch, num_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    m = nn.LogSoftmax()
    n = nn.Softmax()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for j in range(0,epoch):
        for tuple in izip(enumerate(val_loader_s1), enumerate(val_loader)):
            i1, (input_s1, target_s1) = tuple[0]
            i2, (input_v, target_v) = tuple[1]

            # input: 256x3x224x224, target: 256
            input_var_s1 = torch.autograd.Variable(input_s1)
            input_var_v = torch.autograd.Variable(input_v)
            input_var_s1.requires_grad = True
            input_var_v.requires_grad = True
            input = torch.cat((input_var_s1, input_var_v), 0)

            target_var_s1 = torch.autograd.Variable(target_s1)
            target_var_t = torch.autograd.Variable(target_v)
            target_var_s1 = target_var_s1.cuda()
            target_var_t = target_var_t.cuda()
            target = torch.cat((target_s1, target_v), 0)

            # compute output
            output = model(input, alpha1, alpha2, alpha3)

            output_s = output.narrow(0, 0, split)
            output_v = output.narrow(0, split, args.batch_size - split)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output_v.data, target_v.cuda(), topk=(1, 5))
            top1.update(prec1[0], input_v.size(0))
            top5.update(prec5[0], input_v.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i1 % args.print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(j,
                       i1, len(val_loader), top1=top1, top5=top5))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.1 ** (epoch // 54))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        # type: () -> object
        return min(len(d) for d in self.datasets)

if __name__ == '__main__':
    main()
    print("done")
