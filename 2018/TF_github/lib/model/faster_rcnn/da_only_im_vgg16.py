# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.da_only_im_faster_rcnn import _da_fasterRCNN
import pdb
import numpy as np
import cv2


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        grad_input = - 0.1 * grad_output.clone()
        return grad_input


class ImageLevelDA(nn.Module):
    def __init__(self):
        super(ImageLevelDA, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2, 1)
        )
        # self.resize = LabelResizeLayer_im()

    def forward(self, feat, label):
        feat = GradReverse.apply(feat)
        feat = self.layers(feat)

        label = LabelResizeLayer_im(feat, label)
        loss = F.cross_entropy(feat, label)
        return loss


def LabelResizeLayer_im(feats, lbs):
    lbs = lbs.data.cpu().numpy()
    lbs_resize = cv2.resize(lbs, (feats.shape[3], feats.shape[2]), interpolation=cv2.INTER_NEAREST)

    gt_blob = np.zeros((1, lbs_resize.shape[0], lbs_resize.shape[1], 1), dtype=np.float32)
    gt_blob[0, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

    channel_swap = (0, 3, 1, 2)
    gt_blob = gt_blob.transpose(channel_swap).astype(int)

    # gt_blob_onehot = np.zeros((gt_blob.shape[0], 2, gt_blob.shape[2], gt_blob.shape[3]))
    #
    # gt_blob_onehot[0, gt_blob[0,:,:]] = 1

    gt_blob = torch.squeeze(Variable(torch.from_numpy(gt_blob).long().cuda(), requires_grad=False), dim=1)
    return gt_blob


class vgg16(_da_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):
        self.model_path = '../data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _da_fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

        # define imge-level adaptation
        self.im_da = ImageLevelDA()

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
