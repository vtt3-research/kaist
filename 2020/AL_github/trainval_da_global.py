# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
  adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.da_global_vgg16 import vgg16
from model.faster_rcnn.da_global_resnet import resnet


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--exp', default='da_global')

  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='voc_clipart', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='res101', type=str)
  parser.add_argument('--total_step', default=70000, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
  parser.add_argument('--save_iter', default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda', default=True,
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

  # config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=50000, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

  # set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

  # resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
  # log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0, batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "voc_clipart":
    args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
    args.imdbval_name = "clipart_all"
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # -------------------------
  # SOURCE DATA
  # -------------------------
  imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(args.imdb_name)
  train_size_s = len(roidb_s)

  print('{:d} roidb entries'.format(len(roidb_s)))

  sampler_batch_s = sampler(train_size_s, args.batch_size)

  dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, args.batch_size,
                           imdb_s.num_classes, training=True)

  dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                             sampler=sampler_batch_s, num_workers=args.num_workers)

  # -------------------------
  # TARGET DATA
  # -------------------------
  imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdbval_name)
  train_size_t = len(roidb_t)

  print('{:d} roidb entries'.format(len(roidb_t)))

  sampler_batch_t = sampler(train_size_t, args.batch_size)

  dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size,
                           imdb_t.num_classes, training=True)

  dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                             sampler=sampler_batch_t, num_workers=args.num_workers)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb_s.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb_s.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb_s.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb_s.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  # tr_momentum = cfg.TRAIN.MOMENTUM
  # tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  if args.resume:
    load_name = os.path.join(output_dir,
                             args.exp + '_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch_s = int(train_size_s / args.batch_size)
  iters_per_epoch_t = int(train_size_t / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter

    logger = SummaryWriter("logs")

  # setting to train mode
  fasterRCNN.train()
  loss_temp = 0
  start = time.time()
  data_iter_s = iter(dataloader_s)
  data_iter_t = iter(dataloader_t)
  for step in range(args.total_step):
    if (step+1) % (args.lr_decay_step) == 0:
      adjust_learning_rate(optimizer, args.lr_decay_gamma)
      lr *= args.lr_decay_gamma

    fasterRCNN.zero_grad()
    optimizer.zero_grad()

    # Domain Label
    valid_label = Variable(torch.ones(args.batch_size).long().cuda())
    fake_label = Variable(torch.zeros(args.batch_size).long().cuda())

    # -------------------------
    # SOURCE DATA
    # -------------------------
    try:
      data_s = next(data_iter_s)
    except StopIteration:
      data_iter_s = iter(dataloader_s)
      data_s = next(data_iter_s)
    im_data_s = Variable(data_s[0].cuda())
    im_info_s = Variable(data_s[1].cuda())
    gt_boxes_s = Variable(data_s[2].cuda())
    num_boxes_s = Variable(data_s[3].cuda())

    # Forward
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, global_da_loss_s, feat_s = fasterRCNN(im_data_s, im_info_s, gt_boxes_s, num_boxes_s, valid_label, True)

    w_s = -1.0 * torch.log(F.softmax(feat_s)[:, 1])

    loss_s = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
           + global_da_loss_s.mean() * 0.5
    loss_s = loss_s * w_s
    loss_temp += loss_s.item()
    loss_s.backward()

    # -------------------------
    # TARGET DATA
    # -------------------------
    try:
      data_t = next(data_iter_t)
    except StopIteration:
      data_iter_t = iter(dataloader_t)
      data_t = next(data_iter_t)
    im_data_t = Variable(data_t[0].cuda())

    # Forward
    _, _, _, \
    _, _, \
    _, _, \
    _, global_da_loss_t, feat_t = fasterRCNN(im_data_t, None, None, None, fake_label, False)

    loss_t = global_da_loss_t.mean() * 0.5
    loss_t.backward()

    # optimizer step
    if args.net == "vgg16":
      clip_gradient(fasterRCNN, 10.)
    optimizer.step()

    if step % args.disp_interval == 0:
      end = time.time()
      if step > 0:
        loss_temp /= (args.disp_interval + 1)

      loss_rpn_cls = rpn_loss_cls.item()
      loss_rpn_box = rpn_loss_box.item()
      loss_rcnn_cls = RCNN_loss_cls.item()
      loss_rcnn_box = RCNN_loss_bbox.item()
      fg_cnt = torch.sum(rois_label.data.ne(0))
      bg_cnt = rois_label.data.numel() - fg_cnt

      print("[session %d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
            % (args.session, step, args.total_step, loss_temp, lr))
      print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
      print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
            % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
      if args.use_tfboard:
        info = {
          'loss': loss_temp,
          'loss_rpn_cls': loss_rpn_cls,
          'loss_rpn_box': loss_rpn_box,
          'loss_rcnn_cls': loss_rcnn_cls,
          'loss_rcnn_box': loss_rcnn_box
        }
        logger.add_scalars("logs_s_{}/losses".format(args.session), info, step)

      loss_temp = 0
      start = time.time()

    if (step+1)%args.save_iter==0:
      save_name = os.path.join(output_dir, args.exp + '_{}_{}.pth'.format(args.session, step+1))
      save_checkpoint({
        'session': args.session,
        'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': args.class_agnostic,
      }, save_name)
      print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
