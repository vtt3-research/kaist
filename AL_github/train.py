import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import *
from itertools import izip
import numpy as np
import argparse
import time
import os
from model import *
from tensorboardX import SummaryWriter

CUDA_LAUNCH_BLOCKING=1
writer = SummaryWriter()

def main():
    global args, best_prec1
    parser = argparse.ArgumentParser(description='PyTorch Active Learning')
    args = parser.parse_args()

    args.arch = 'alexnet'
    args.batch_size = 256
    args.lr = 5e-2
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.epochs = 60
    args.num_classes = 100
    args.sample_per_batch = 10
    args.data = '/home/tk/PycharmProjects/rltest3/data'
    args.print_freq = 1
    args.data_name = 'caltech256'
    args.all_class = True
    args.active = True

    labdir = os.path.join(args.data, 'labeled')
    unlabdir = os.path.join(args.data, 'unlabeled')
    testdir = os.path.join(args.data, 'test')

    model = alexnet(pretrained =True, num_classes = args.num_classes)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    lab_dataset = datasets.ImageFolder(
                labdir,
                transforms.Compose([
                    transforms.Scale(256),
                    transforms.RandomCrop(227),
                    transforms.RandomHorizontalFlip(),

                    transforms.ToTensor(),
                    normalize,
                ]))

    unlab_dataset = datasets.ImageFolder(
                unlabdir,
                transforms.Compose([
                    transforms.Scale(256),
                    transforms.RandomCrop(227),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
    test_dataset = datasets.ImageFolder(
                testdir,
                transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))

    lab_sampler = RandomSampler(lab_dataset)
    unlab_sampler = RandomSampler(unlab_dataset)
    test_sampler = RandomSampler(test_dataset)

    lab_loader = torch.utils.data.DataLoader(
        lab_dataset, batch_size=256, shuffle=True,
        pin_memory=True, sampler=None, drop_last=False)

    unlab_loader = torch.utils.data.DataLoader(
        unlab_dataset, batch_size=256, shuffle=False,
        pin_memory=True, sampler=None, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=None,
        pin_memory=True, sampler=None, drop_last=False)


    # fine-tuning with labeled data
    print 'fine-tuning model with labeled data'
    for i in range(0,10):
        print i+1, '/10'
        pretraining(lab_loader, model, criterion, optimizer, epoch=10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001
    pretraining(lab_loader, model, criterion, optimizer, epoch=10)
    validate(test_loader, model, criterion, optimizer,-1, epoch=1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001


    # AL / Random
    for i in range(0,15):
        delta = 0.05 - 0.00033 * i
        print '* ', i, 'th sampling'

        if args.active:
            sample, pseudo_lab, lab_dataset, unlab_dataset = Sampling(model, delta, lab_dataset, unlab_dataset, lab_sampler, unlab_sampler, labdir, 'LC')
            lab_sampler = RandomSampler(lab_dataset)
            lab_4loader = torch.utils.data.DataLoader(
                lab_dataset, batch_size=256, shuffle=True,
                pin_memory=True, sampler=None, drop_last=False)

            unlab_sampler = RandomSampler(unlab_dataset)
            unlab_loader = torch.utils.data.DataLoader(
                unlab_dataset, batch_size=256, shuffle=True,
                pin_memory=True, sampler=None, drop_last=False)

            if len(pseudo_lab) > 0:
                print '* low confidence dataset is sampled'
                print '* sampled', len(pseudo_lab)
        else:
            sample, pseudo_lab, lab_dataset, unlab_dataset = Sampling(model, delta, lab_dataset, unlab_dataset, lab_sampler, unlab_sampler, labdir, 'R')
            lab_sampler = RandomSampler(lab_dataset)
            lab_loader = torch.utils.data.DataLoader(
                lab_dataset, batch_size=256, shuffle=True,
                pin_memory=True, sampler=None, drop_last=False)

            unlab_sampler = RandomSampler(unlab_dataset)
            unlab_loader = torch.utils.data.DataLoader(
                unlab_dataset, batch_size=256, shuffle=True,
                pin_memory=True, sampler=None, drop_last=False)

            if len(pseudo_lab) > 0:
                print '* low confidence dataset is sampled'
                print '* sampled', len(pseudo_lab)

        # High confidence hidden dataset training
        sample, pseudo_lab, hid_data = Sampling(model, delta, lab_dataset, unlab_dataset, lab_sampler, unlab_sampler, labdir, 'HC')

        if len(hid_data.imgs) > 0:
            lab_hid_dataset = hid_data
            lab_hid_dataset.imgs = lab_hid_dataset.imgs + lab_dataset.imgs
        else:
            lab_hid_dataset = lab_dataset
        lab_hid_loader = torch.utils.data.DataLoader(
            lab_hid_dataset, batch_size=256, shuffle=True,
            pin_memory=True, sampler=None, drop_last=False)

        if len(pseudo_lab) > 0:
            print '* ', len(hid_data), 'labeled + hidden dataset is sampled'
            # train(lab_hidden_loader, model, criterion, optimizer, epoch=1)
            print '* sample training finished'
        lab_hid_loader = torch.utils.data.DataLoader(
            lab_dataset, batch_size=256, shuffle=True,
            pin_memory=True, sampler=None, drop_last=False)

        for i in range(0,5):
            train(lab_hid_loader, model, criterion, optimizer, epoch=5)
            validate(test_loader, model, criterion, optimizer, i, epoch=1)
        print '* test accuracy'
        validate(test_loader, model, criterion, optimizer, i, epoch=1)


def pretraining(loader, model, criterion, optimizer, epoch):
    m = nn.Softmax()
    n = nn.LogSoftmax()
    model.train()
    top1 = AverageMeter()
    top5 = AverageMeter()

    freeze = 0
    for param in model.parameters():
        if freeze < 10:
            param.requires_grad = False
            freeze += 1

    for j in range(0,epoch):
        for i, (input,target) in enumerate(loader):
            input_var = Variable(input, requires_grad=True).cuda(async=True)
            target_var = Variable(target, requires_grad=True).cuda(async=True)

            output = model(input_var)
            loss = criterion(m(output),target_var)
            # loss = - torch.mean(torch.sum(m(output) * n(output), 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1, prec5 = accuracy(output.data, target.cuda(), topk=(1,5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    j, i, len(loader),  top1=top1, top5=top5))
        top1 = AverageMeter()
        top5 = AverageMeter()

def Sampling(model, delta, lab_dataset, unlab_dataset, lab_sampler, unlab_sampler, any_dir, option):
    m = nn.Softmax()
    n = nn.LogSoftmax()

    sample = []
    sample_lab = []
    if option == 'HC':
        count = 0
        total = 0
        batch = []
        label = []
        en_val = []
        en_ind = []

        req_ind = []
        sampler = list(unlab_sampler)
        unlab_loader = torch.utils.data.DataLoader(
            unlab_dataset, batch_size=256, shuffle=False,
            pin_memory=True, sampler=sampler, drop_last=False)
        print len(lab_dataset), len(unlab_dataset)
        hid_dataset = datasets.ImageFolder(
            any_dir,
            transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(227),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
        hid_dataset.imgs = []

        for i, (input, target) in enumerate(unlab_loader):
            input_var = Variable(input, requires_grad=False).cuda(async=True)
            output = model(input_var)
            value, index = m(output).max(1)
            en = -torch.sum(m(output) * n(output), 1)

            index_cpu = index.data
            for j in range(0,len(en)):
                if en[j] < delta:
                    pseudo_label = index_cpu[j]
                    batch.append(input[j,:,:,:])
                    label.append(pseudo_label)

                    en_val.append(en[j])
                    en_ind.append(256 * i + j)

                    count += 1
                    total += 1
                if count == 255:
                    count = 0
                    batch = []
                    label = []
            sample.append(batch)
            sample_lab.append(label)
        L = [ (-en_val[i], en_ind[i]) for i in xrange(len(en_val))]
        L.sort()
        print 'total candidates', len(en_ind)
        if len(en_val) > 1000:
            en_ind = en_ind[:1000]
        for l in range(0,len(en_ind)):
            (hid_dataset.imgs).append(unlab_dataset.imgs[list(sampler)[en_ind[l]]])

        if total > 1000:
            print 'hidden data num exceeds 1000'
        print len(lab_dataset.imgs)
        return sample, sample_lab, hid_dataset

    elif option == 'LC':
        count = 0
        batch = []
        label = []
        K = 100
        k = 14
        maxk = max((k, ))
        topk = (1,)

        req_ind = []
        sampler = list(unlab_sampler)
        unlab_loader = torch.utils.data.DataLoader(
            unlab_dataset, batch_size=256, shuffle=False,
            pin_memory=True, sampler=sampler, drop_last=False)
        print len(lab_dataset), len(unlab_dataset)
        for i, (input, target) in enumerate(unlab_loader):
            input_var = Variable(input, requires_grad=False).cuda(async=True)
            output = model(input_var)
            value, index = m(output).max(1)
            _, botk_ind = (-value).topk(maxk)
            botk_ind = botk_ind.data

            for l in range(0,len(botk_ind)):
                j = botk_ind[l]

                # request to oracle
                pseudo_label = target[j]
                batch.append(input[j,:,:,:])
                label.append(pseudo_label)

                (lab_dataset.imgs).append(unlab_dataset.imgs[list(sampler)[256 * i + j]])
                req_ind.append(256 * i + j)
                count += 1

                if count == 255:
                    count = 0
                    batch = []
                    label = []

        req_num = len(req_ind)
        req_ind.sort()
        for l in range(0, req_num):
            (unlab_dataset.imgs).pop(req_ind[req_num - l - 1])

        print 'total', i+1, 'batches', len(lab_dataset), len(unlab_dataset)
        return sample, sample_lab, lab_dataset, unlab_dataset

    elif option == 'R':
        count = 0
        batch = []
        label = []
        K = 1000
        k = 14

        req_ind = []
        sampler = list(unlab_sampler)
        unlab_loader = torch.utils.data.DataLoader(
            unlab_dataset, batch_size=256, shuffle=False,
            pin_memory=True, sampler=sampler, drop_last=False)
        print len(lab_dataset), len(unlab_dataset)
        for i, (input, target) in enumerate(unlab_loader):
            input_var = Variable(input, requires_grad=False).cuda(async=True)
            output = model(input_var)
            value, index = m(output).max(1)
            rand_ind = torch.randperm(len(target))
            for l in range(0,min(len(rand_ind),k)):
                j = rand_ind[l]

                # request to oracle
                pseudo_label = target[j]
                batch.append(input[j,:,:,:])
                label.append(pseudo_label)

                (lab_dataset.imgs).append(unlab_dataset.imgs[list(sampler)[256 * i + j]])
                req_ind.append(256 * i + j)
                count += 1

                if count == 255:
                    count = 0
                    batch = []
                    label = []
        req_num = len(req_ind)
        req_ind.sort()
        for l in range(0, req_num):
            (unlab_dataset.imgs).pop(req_ind[req_num - l - 1])

        return sample, sample_lab, lab_dataset, unlab_dataset


def train(train_loader, model, criterion, optimizer, epoch):
    m = nn.LogSoftmax()
    n = nn.Softmax()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for e in range(0,epoch):
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            input_var = Variable(input,requires_grad=True).cuda()
            target_var = Variable(target, requires_grad=True).cuda()
            output = model(input_var)
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target.cuda(), topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
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
                    e, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

def sample_train(sample,label, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i in range(0, len(label)):
        data_time.update(time.time() - end)

        input = sample[i]
        target = label[i]
        input_var = Variable(input,requires_grad=True).cuda()
        target_var = target
        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def validate(train_loader, model, criterion, optimizer, writer_epoch_axis, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for e in range(0,epoch):
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            input_var = Variable(input, requires_grad=False).cuda()
            target_var = Variable(target, requires_grad=False).cuda()
            output = model(input_var)
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target.cuda(), topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    e, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                batch_time.update(time.time() - end)
                end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    writer.add_scalars('data/test_accuracy', {'top1': top1.avg, 'top5': top5.avg}, writer_epoch_axis + 1)


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
        param_group['lr'] = param_group['lr'] * (0.1 ** (epoch // 75))


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


