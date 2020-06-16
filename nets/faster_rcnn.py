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
from utils.util import loc2box, non_maximum_suppression
import numpy as np
from configs.config import class_num

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
        self.rpn = RPN()
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
        # rpn_locs维度为: [batch_size, w, h, 4*k], 类型是pytorch的张量
        # rpn_scores维度为: [batch_size, w, h, k], 类型是pytorch的张量
        # anchors维度为: [batch_size, w*h*k, 4], 类型是numpy数组
        # rois维度为: [w*h*k ,4]
        rpn_locs, rpn_scores, anchors, rois = self.rpn(h, img_size)
        # gt_anchor_locs维度为: [anchors_num, 4], gt_anchor_labels维度为:[anchors_num, 1]
        gt_anchor_locs, gt_anchor_labels = self.anchor_target_creator(gt_boxes[0].detach().cpu().numpy(),
                                                                      anchors,
                                                                      img_size)

        # ----------------part 3: roi采样部分----------------------------
        sample_rois, gt_roi_labels, gt_roi_locs = self.sample_rois(rois,
                                                                   gt_boxes[0].detach().cpu().numpy(),
                                                                   labels[0].detach().cpu().numpy())

        # ---------------part 4: fast rcnn(roi)部分(output_2)------------
        # roi_cls_locs维度为: [batch_size, 4], roi_scores维度为:[batch_size, 1]
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

    @torch.no_grad()
    def predict(self, x):
        # 设置为测试模式, 改变rpn网络中n_post_nms的阈值为300
        self.eval()

        # -----------------part 1: feature 提取部分----------------------
        h = self.extractor(x)
        img_size = (x.size(2), x.size(3))

        # ----------------------part 2: rpn部分--------------------------
        rpn_locs, rpn_socres, anchors, rois = self.rpn(h, img_size)

        # ------------------part 3: fast rcnn(roi)部分-------------------
        # 先经过Roi pooling层, 在经过两个全连接层
        roi_cls_locs, roi_scores = self.fast_rcnn(h, np.asarray(rois))
        n_sample = roi_cls_locs.shape[0]

        # --------------------part 4:boxes生成部分-----------------------
        roi_cls_locs = roi_cls_locs.view(n_sample, -1, 4)
        rois = torch.from_numpy(rois).to(device)
        rois = rois.view(-1, 1, 4).expand_as(roi_cls_locs)
        boxes = loc2box(rois.cpu().numpy().reshape((-1, 4)), roi_cls_locs.cpu().numpy().reshape((-1, 4)))
        boxes = torch.from_numpy(boxes).to(device)
        # 修剪boxes中的坐标, 使其落在图片内
        boxes[:, [0, 2]] = (boxes[:, [0, 2]]).clamp(min=0, max=img_size[0])
        boxes[:, [1, 3]] = (boxes[:, [1, 3]]).clamp(min=0, max=img_size[1])
        boxes = boxes.view(n_sample, -1)

        # roi_scores转换为概率, prob维度为[rois_num, 7]
        prob = F.softmax(roi_scores, dim=1)

        # ----------------part 5:筛选环节------------------------
        raw_boxes = boxes.cpu().numpy()
        raw_prob = prob.cpu().numpy()
        final_boxes, labels, scores = self._suppress(raw_boxes, raw_prob)
        self.train()
        return final_boxes, labels, scores

    def _suppress(self, raw_boxes, raw_prob):
        # print(raw_prob.shape)
        score_thresh = 0.7
        nms_thresh = 0.3
        n_class = 7
        box = list()
        label = list()
        score = list()

        for i in range(1, class_num):
            box_i = raw_boxes.reshape((-1, n_class, 4))
            box_i = box_i[:, i, :]  # 维度为: [rois_num, k, 4]
            prob_i = raw_prob[:, i]  # 维度为: [rois_num]
            mask = prob_i > score_thresh
            box_i = box_i[mask]
            prob_i = prob_i[mask]
            order = prob_i.argsort()[::-1]
            # 按照score值从大到小进行排序
            box_i = box_i[order]

            box_i_after_nms, keep = non_maximum_suppression(box_i, nms_thresh)
            box.append(box_i_after_nms)

            label_i = (i - 1) * np.ones((len(keep),))
            label.append(label_i)
            score.append(prob_i[keep])

        box = np.concatenate(box, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return box, label, score


if __name__ == '__main__':
    path = '../pre_model_weights/vgg16-397923af.pth'
    faster_rcnn = FasterRCNN(path)
    # faster_rcnn.predict(torch.ones((1, 3, 224, 224)))
