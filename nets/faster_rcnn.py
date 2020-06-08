import torch
from torch import nn
import torch.nn.functional as F
from nets.vgg16 import decom_VGG16
from nets.rpn import RPN
from nets.anchor_target_creator import AnchorTargetCreator
from nets.proposal_target_creator import ProposalTargetCreator
from nets.fast_rcnn import FastRCNN
from utils.util import loc_loss
from collections import namedtuple
import torch
from configs.config import class_num, in_channels

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

device = torch.device("cuda")


class FasterRCNN(nn.Module):
    def __init__(self, path):
        super(FasterRCNN, self).__init__()

        self.extractor, classifier = decom_VGG16(path)
        self.rpn = RPN(in_channels=in_channels)
        self.anchor_target_creator = AnchorTargetCreator()
        self.sample_rois = ProposalTargetCreator()

        self.fast_rcnn = FastRCNN(n_class=class_num, roi_size=7, spatial_scale=1. / 16, classifier=classifier)
        # 系数,用来计算l1_smooth_loss
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

    def forward(self, x, gt_boxes, labels):
        # -----------------part 1: feature 提取部分----------------------
        h = self.extractor(x)

        # -----------------part 2: rpn部分(output_1)----------------------
        img_size = (x.size(2), x.size(3))
        rpn_locs, rpn_scores, anchors, rois = self.rpn(h, img_size)
        gt_anchor_locs, gt_anchor_labels = self.anchor_target_creator(gt_boxes[0].detach().cpu().numpy(),
                                                                      anchors,
                                                                      img_size)

        # ----------------part 3: roi采样部分----------------------------
        sample_rois, gt_roi_labels, gt_roi_locs = self.sample_rois(rois,
                                                                   gt_boxes[0].detach().cpu().numpy(),
                                                                   labels[0].detach().cpu().numpy())

        # ---------------part 4: fast rcnn(roi)部分(output_2)------------
        roi_cls_locs, roi_scores = self.fast_rcnn(h, sample_rois)

        # RPN LOSS
        gt_anchor_locs = torch.from_numpy(gt_anchor_locs).to(device)
        gt_anchor_labels = torch.from_numpy(gt_anchor_labels).long().to(device)
        rpn_cls_loss = F.cross_entropy(rpn_scores[0], gt_anchor_labels, ignore_index=-1)  # label值为-1的不参与loss值的计算
        rpn_loc_loss = loc_loss(rpn_locs[0], gt_anchor_locs, gt_anchor_labels, self.rpn_sigma)

        # ROI LOSS
        gt_roi_labels = torch.from_numpy(gt_roi_labels).long().to(device)
        gt_roi_locs = torch.from_numpy(gt_roi_locs).float().to(device)
        roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels)
        n_sample = roi_cls_locs.shape[0]  # batch_size
        roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)
        roi_locs = roi_cls_locs[torch.arange(0, n_sample).long(), gt_roi_labels]
        roi_loc_loss = loc_loss(roi_locs.contiguous(), gt_roi_locs, gt_roi_labels, self.roi_sigma)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)


if __name__ == '__main__':
    path = '../pre_model_weights/vgg16-397923af.pth'
    faster_rcnn = FasterRCNN(path)
