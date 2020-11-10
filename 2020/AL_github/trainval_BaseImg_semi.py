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
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.BaseImg_semi_vgg16 import vgg16


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--exp', default='baseimg_semi_try3')
  parser.add_argument('--source',
                      help='source dataset',
                      default="pascal_voc", type=str)
  parser.add_argument('--target',
                      help='target dataset',
                      default="clipart", type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=3, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

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

# config optimization7
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=34, type=int)
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
                      default=9, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=10022, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  # SOURCE
  if args.source == "pascal_voc":
      args.imdb_source = "voc_2007_trainval+voc_2012_trainval"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  # TARGET
  if args.target == "clipart":
      args.imdb_target = "clipart_all_without_label"
      args.imdb_guide = "clipart_all_with_label"
      args.imdb_test = "clipart_all_without_label"
  elif args.target == "watercolor":
      args.imdb_target = "watercolor_train_without_label"
      args.imdb_guide = "watercolor_train_with_label"
      args.imdb_test = "watercolor_test"
  elif args.target == "comic":
      args.imdb_target = "comic_train_without_label"
      args.imdb_guide = "comic_train_with_label"
      args.imdb_test = "comic_test"

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  # SOURCE
  imdb_s, roidb_s, ratio_list_s, ratio_index_s = combined_roidb(args.imdb_source)
  train_size_s = len(roidb_s)

  print('%s: %d roidb entries' % (imdb_s.name, len(roidb_s)))

  sampler_batch_s = sampler(train_size_s, args.batch_size)

  dataset_s = roibatchLoader(roidb_s, ratio_list_s, ratio_index_s, 8, \
                           imdb_s.num_classes, training=True)

  dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=8,
                            sampler=sampler_batch_s, num_workers=args.num_workers)

  # TARGET
  imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_target)
  train_size_t = len(roidb_t)

  print('%s: %d roidb entries' % (imdb_t.name, len(roidb_t)))

  sampler_batch_t = sampler(train_size_t, args.batch_size)

  dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, 8, \
                           imdb_t.num_classes, training=True)

  dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=8,
                            sampler=sampler_batch_t, num_workers=args.num_workers)

  # GUIDE
  imdb_g, roidb_g, ratio_list_g, ratio_index_g = combined_roidb(args.imdb_guide)
  train_size_g = len(roidb_g)

  print('%s: %d roidb entries' % (imdb_g.name, len(roidb_g)))

  sampler_batch_g = sampler(train_size_g, args.batch_size)

  dataset_g = roibatchLoader(roidb_g, ratio_list_g, ratio_index_g, 1, \
                           imdb_g.num_classes, training=True)

  dataloader_g = torch.utils.data.DataLoader(dataset_g, batch_size=1,
                                           sampler=sampler_batch_g, num_workers=args.num_workers)

  output_dir = args.save_dir + "/" + args.net + "/" + args.source + "_" + args.target
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  dc_label = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    dc_label = dc_label.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  dc_label = Variable(dc_label)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb_s.classes, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  params2 = []
  params3 = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'D_img' in key:
        if 'bias' in key:
          params += [{'params': [value], 'lr': lr*10 *(cfg.TRAIN.DOUBLE_BIAS + 1), \
                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else :
          params += [{'params':[value], 'lr':lr*10, 'weight_decay':cfg.TRAIN.WEIGHT_DECAY}]

      else:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    # optimizer2 = torch.optim.SGD(params2, momentum=cfg.TRAIN.MOMENTUM)
    # optimizer3 = torch.optim.SGD(params3, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch_s = int(train_size_s / args.batch_size)
  iters_per_epoch_t = int(train_size_t / args.batch_size)
  iters_per_epoch_g = int(train_size_g / args.batch_size)
  first = 1
  optimizer.zero_grad()
  fasterRCNN.zero_grad()
  count = 0
  train_end = False
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    data_iter_s = iter(dataloader_s)
    data_iter_t = iter(dataloader_t)
    data_iter_g = iter(dataloader_g)
    for step in range(iters_per_epoch_s):
      if (count+1) % 50000 == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma
      count += 1

      if step % iters_per_epoch_t == 0:
          data_iter_t = iter(dataloader_t)

      if step % iters_per_epoch_g == 0:
          data_iter_g = iter(dataloader_g)

      # SOURCE
      data_s = next(data_iter_s)
      im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
      im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
      gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
      num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

      need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
      dc_label_tmp =  torch.from_numpy(np.ones((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, RCNN_loss_img, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                             need_backprop=need_backprop, dc_label=dc_label)
      loss_s = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
           + RCNN_loss_img.mean() #+ RCNN_loss_ins.mean()
      loss_temp += loss_s.data.item()

      loss_s.backward()

      # TARGET
      data_t = next(data_iter_t)
      im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
      im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
      gt_boxes.data.resize_(data_t[2].size()).copy_(data_t[2])
      num_boxes.data.resize_(data_t[3].size()).copy_(data_t[3])

      need_backprop = torch.from_numpy(np.zeros((1,), dtype=np.float32))
      dc_label_tmp =  torch.from_numpy(np.zeros((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

      rois2, cls_prob2, bbox_pred2, \
      rpn_loss_cls2, rpn_loss_box2, \
      RCNN_loss_cls2, RCNN_loss_bbox2, \
      rois_label2, RCNN_loss_img2, ssda_mme = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                               need_backprop=need_backprop, dc_label=dc_label)
      loss_t =  RCNN_loss_img2.mean() + ssda_mme.mean() # + RCNN_loss_ins2.mean()
      loss_temp += loss_t.item()

      loss_t.backward()

      # GUIDE
      if step % 10 == 0:
          data_g = next(data_iter_g)
          im_data.data.resize_(data_g[0].size()).copy_(data_g[0])
          im_info.data.resize_(data_g[1].size()).copy_(data_g[1])
          gt_boxes.data.resize_(data_g[2].size()).copy_(data_g[2])
          num_boxes.data.resize_(data_g[3].size()).copy_(data_g[3])

          need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))

          # need_G_img = torch.from_numpy(np.zeros((1,), dtype=np.float32))
          dc_label_tmp = torch.from_numpy(np.zeros((2000, 1), dtype=np.float32))
          dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

          rois3, cls_prob3, bbox_pred3, \
          rpn_loss_cls3, rpn_loss_box3, \
          RCNN_loss_cls3, RCNN_loss_bbox3, \
          rois_label3, RCNN_loss_img3, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                                   need_backprop=need_backprop, dc_label=dc_label)
          loss_g = rpn_loss_cls3.mean() + rpn_loss_box3.mean() \
                 + RCNN_loss_cls3.mean() + RCNN_loss_bbox3.mean() \
                 # + RCNN_loss_img3.mean()  # + RCNN_loss_ins2.mean()
          loss_temp += loss_g.item()

          loss_g.backward()

      optimizer.step()
      optimizer.zero_grad()
      fasterRCNN.zero_grad()



      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= args.disp_interval

        loss_rpn_cls = rpn_loss_cls.item()
        loss_rpn_box = rpn_loss_box.item()
        loss_rcnn_cls = RCNN_loss_cls.item()
        loss_rcnn_box = RCNN_loss_bbox.item()
        loss_rcnn_img = RCNN_loss_img.item()
        # loss_rcnn_ins = RCNN_loss_ins.mean().data[0]
        loss_rpn_cls2 = rpn_loss_cls2.mean().item()
        loss_rpn_box2 = rpn_loss_box2.mean().item()
        loss_rcnn_cls2 = RCNN_loss_cls2.mean().item()
        loss_rcnn_box2 = RCNN_loss_bbox2.mean().item()
        loss_rcnn_img2 = RCNN_loss_img2.item()

        if step % 10 == 0:
            # loss_rcnn_ins2 = RCNN_loss_ins2.mean().data[0]
            loss_rpn_cls3 = rpn_loss_cls3.mean().item()
            loss_rpn_box3 = rpn_loss_box3.mean().item()
            loss_rcnn_cls3 = RCNN_loss_cls3.mean().item()
            loss_rcnn_box3 = RCNN_loss_bbox3.mean().item()
            loss_rcnn_img3 = RCNN_loss_img3.item()

        loss_ssda_mme = ssda_mme.mean().item()

        fg_cnt = torch.sum(rois_label.data.ne(0))
        bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch_s, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, source_rcnn_img: %4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_img))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, target_rcnn_img: %4f" \
            % (loss_rpn_cls2, loss_rpn_box2, loss_rcnn_cls2, loss_rcnn_box2, loss_rcnn_img2))

        if step % 10 == 0:
            print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, sliket_rcnn_img: %4f" \
              % (loss_rpn_cls3, loss_rpn_box3, loss_rcnn_cls3, loss_rcnn_box3, loss_rcnn_img3 ))
        print('ssda_mme: %.4f' % loss_ssda_mme)

        loss_temp = 0
        start = time.time()
        if count == 70000:
          train_end = True

      if (count+1) % 2000 == 0:
        save_name = os.path.join(output_dir,
                               '{}_{}_{}.pth'.format(
                                   args.exp, args.session, count + 1))
        save_checkpoint({
          'session': args.session,
          # 'epoch': epoch + 1,
          'model': fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)
