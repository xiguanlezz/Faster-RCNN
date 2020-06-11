import numpy as np
from utils.util import loc2box, non_maximum_suppression


class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        """
        :param parent_model: 区分是training_model还是testing_model
        :param nms_thresh: 非极大值抑制的阈值
        :param n_train_pre_nms: 训练时NMS之前的boxes的数量
        :param n_train_post_nms: 训练时NMS之后的boxes的数量
        :param n_test_pre_nms: 测试时NMS之前的数量
        :param n_test_post_nms: 测试时NMS之后的数量
        :param min_size: 生成一个roi所需的目标的最小高度, 防止Roi pooling层切割后维度降为0
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
        function description: 通过rpn网络输出的locs来校正先验框anchors的位置并完成NMS, 返回固定数量的rois

        :param locs: rpn网络中的1x1卷积的一个输出, 维度为[w*h*k, 4]
        :param scores: rpn网络中的1x1卷积的另一个输出, 维度为:[w*h*k, 2]
        :param anchors: 先验框
        :param img_size: 输入整个Faster-RCNN网络的图片尺寸
        :return:
            roi_after_nms: 通过rpn网络输出的locs来校正先验框anchors的位置并完成NMS之后的rois
        """
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # 根据rpn_locs微调先验框即将anchors转化为rois
        roi = loc2box(anchors, locs)

        # 防止建议框即rois超出图像边缘
        roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, img_size[0])  # 对X轴剪切
        roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, img_size[1])  # 对Y轴剪切

        # 去除高或宽<min_size的rois, 防止Roi pooling层切割后维度降为0
        min_size = self.min_size
        roi_width = roi[:, 2] - roi[:, 0]
        roi_height = roi[:, 3] - roi[:, 1]
        keep = np.where((roi_width >= min_size) & (roi_height >= min_size))[0]  # 得到满足条件的行index
        roi = roi[keep, :]

        scores = scores[:, 1]
        scores = scores[keep]
        # argsort()函数得到的是从小到大的索引, x[start:end:span]中如果span<0则逆序遍历; 如果span>0则顺序遍历
        order = scores.argsort()[::-1]  # 对roi通过rpn的scores进行排序, 得到scores的下降排列的坐标
        # 保留分数排在前面的n_pre_nms个rois
        order = order[: n_pre_nms]
        roi = roi[order, :]

        # 非极大值抑制
        roi_after_nms, _ = non_maximum_suppression(roi, thresh=self.nms_thresh)
        # NMS之后保留分数排在前面的n_post_nms个rois
        roi_after_nms = roi_after_nms[:n_post_nms]

        return roi_after_nms
