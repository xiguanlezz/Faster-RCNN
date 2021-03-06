import numpy as np
from utils.util import calculate_iou, box2loc


class ProposalTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        """
        function description: 采样128正负样本个传入FastRCNN的网络

        :param n_sample: 需要采样的数量
        :param pos_ratio: 正样本比例
        :param pos_iou_thresh: 正样本阈值
        :param neg_iou_thresh_hi: 负样本最大阈值
        :param neg_iou_thresh_lo: 负样本最低阈值
        :return:
            sample_rois: 采样后的感兴趣区域
            gt_roi_labels: boxes的标签
            gt_roi_locs: sample_rois和boxes的线性回归值
        """
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self,
                 rois,
                 boxes,
                 labels,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """
        function description: 得到采样后的rois, 及其对应的labels和回归值

        :param rois: rpn输入的rois
        :param boxes: 一幅图的位置标注
        :param labels: 一幅图的类别标注
        :param loc_normalize_mean: 均值
        :param loc_normalize_std: 标准差
        :return:
        """
        n_bbox, _ = boxes.shape

        # 取到正样本的个数(四舍五入)
        pos_num = np.round(self.n_sample * self.pos_ratio)

        ious = calculate_iou(rois, boxes)
        gt_assignment = ious.argmax(axis=1)  # 返回维度为[rois_num]
        max_iou = ious.max(axis=1)

        gt_roi_labels = labels[gt_assignment]  # 返回维度为[rois_num]

        # 筛选出其中iou满足阈值的部分
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_num_for_this_image = int(min(pos_num, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_num_for_this_image, replace=False)
        # 筛选出其中iou不满足阈值的部分
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_num = self.n_sample - pos_num_for_this_image
        neg_num_for_this_image = int(min(neg_index.size, neg_num))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_num_for_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_roi_labels = gt_roi_labels[keep_index]
        gt_roi_labels[pos_num_for_this_image:] = 0  # 背景标记为0, pos_num_for_this_image及之后的索引都标为0
        sample_rois = rois[keep_index]

        gt_roi_locs = box2loc(sample_rois, boxes[gt_assignment[keep_index]])

        return sample_rois, gt_roi_labels, gt_roi_locs
