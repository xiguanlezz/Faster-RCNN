import numpy as np
from utils.util import loc2box, non_maximum_suppression


class ProposalCreator():
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        """
        function description: 通过rpn网络的locs值来初步校正先验框anchors为proposals, 并在nms之后保留固定数量的roi用于训练

        :param parent_model: 区分是training_model还是testing_model
        :param nms_thresh: 非极大值抑制的阈值
        :param n_train_pre_nms: 训练时nms之前的boxes的数量
        :param n_train_post_nms: 训练时nms之后的boxes的数量
        :param n_test_pre_nms: 测试时nms之前的数量
        :param n_test_post_nms: 测试时nms之后的数量
        :param min_size: 生成一个proposal所需的目标的最小高度
        :return:
        """
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, locs, scores, anchors, img_size):
        """
        function description: 返回一定的rois

        :param locs: rpn网络中的1x1卷积的一个输出
        :param scores: rpn网络中的1x1卷积的另一个输出
        :param anchors: 先验框
        :param img_size: 宽高
        :return:
        """
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 将先验框转化为proposals
        roi = loc2box(anchors, locs)

        # 防止建议框即proposals超出图像边缘
        roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, img_size[0])  # 对X轴剪切
        roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, img_size[1])  # 对Y轴剪切

        # 去除高或宽<min_size的roi
        min_size = self.min_size
        roi_width = roi[:, 2] - roi[:, 0]
        roi_height = roi[:, 3] - roi[:, 1]
        keep = np.where((roi_width >= min_size) & (roi_height >= min_size))[0]  # 得到满足条件的行index
        roi = roi[keep, :]

        scores = scores[:, 1]
        scores = scores[keep]
        # 对roi通过rpn的scores进行排序, 得到scores的下降排列的坐标
        order = scores.argsort()[::-1]
        # 保留固定数量 训练 12000
        order = order[: n_pre_nms]
        roi = roi[order, :]

        # 非极大值抑制
        roi_after_nms = non_maximum_suppression(roi, thresh=self.nms_thresh)
        # 保留固定数量 训练2000
        roi_after_nms = roi_after_nms[:n_post_nms]

        return roi_after_nms
