import numpy as np
from utils.util import calculate_iou, get_inside_index, box2loc


class AnchorTargetCreator():
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        """
        function description: AnchorTargetCreator构造函数

        :param n_sample: 256, target的总数量
        :param pos_iou_thresh: 和boxes的iou的阈值，超过此值为"正"样本, label会置为1
        :param neg_iou_thresh: 和boxes的iou的阈值，低于此之为"负"样本, label会置为0
        :param pos_ratio: target总数量中"正"样本的比例
        """
        self.n_sample = n_sample
        # 正样本iou的阈值
        self.pos_iou_thresh = pos_iou_thresh
        # 负样本iou的阈值
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio  # target总数量中"正"样本,如果正样本数量不足,则填充负样本

    def __call__(self, boxes, anchors, img_size):
        """
        function description:

        :param boxes: 图片中真实框左上角和右下角的坐标, 维度: [boxes_num, 4]
        :param anchors: RPN网络中输入的anchors坐标, 维度: [anchors_num, 4]
        :param img_size: 原图的大小, 用来过滤掉出界的anchors
        :return:
            anchor_locs: 最终的坐标, 维度为[inside_anchors_num ,4]
            anchor_labels: 最终的标签, 维度为[inside_anchors_num]
        """
        img_width, img_height = img_size

        inside_index = get_inside_index(anchors, img_width, img_height)
        # 根据index取到在图片内部的anchors
        inside_anchors = anchors[inside_index]
        # 返回维度都为[inside_anchors_num]的每个先验框对应的iou最大的真实框的索引及打好的标签
        argmax_ious, labels = self._create_label(inside_anchors, boxes)

        # 计算inside_anchors和对应iou最大的boxes的回归值
        locas = box2loc(inside_anchors, boxes[argmax_ious])

        # 把inside_anchors重新展开回原来所有的anchors
        anchor_labels = np.empty((len(anchors),), dtype=labels.dtype)
        anchor_labels.fill(-1)
        anchor_labels[inside_index] = labels
        # 利用broadcast重新展开locs
        anchor_locs = np.empty((len(anchors),) + anchors.shape[1:], dtype=locas.dtype)
        anchor_locs.fill(0)
        anchor_locs[inside_index, :] = locas
        return anchor_locs, anchor_labels

    def _calculate_iou(self, inside_anchors, boxes):
        """
        function description: 从二位iou张量中获得每个先验框对应的iou最大的真实框的索引以及iou的值
                              以及每个真实框对应的iou最大的先验框的索引???

        :param inside_anchors: 在图片内的先验框(anchors)
        :param boxes: 图片中的真实框
        :return:

        """
        # 第一个维度是先验框的个数(inside_anchors_num), 第二个维度是真实框的个数(boxes_num)
        ious = calculate_iou(inside_anchors, boxes)

        argmax_ious = ious.argmax(axis=1)
        # 取到每个先验框对应的真实框最大的iou
        max_ious = ious[np.arange(len(inside_anchors)), argmax_ious]

        gt_argmax_ious = ious.argmax(axis=0)
        # 取到每个真实框对应的先验框最大的iou
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]

        # 获得第一维的索引
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, inside_anchors, boxes):
        """
        function description: 为每个inside_anchors创建一个label, 其中1表示正样本, 0表示负样本, -1则忽略

        :param inside_anchors: 在图片内的先验框(anchors), 维度为: [inside_anchors_num, 4]
        :param boxes: 图片中的真实标注框, 维度为: [boxes_num, 4]
        :return:
            argmax_ious: 每个先验框对应的iou最大的真实框的索引, 维度为: [inside_anchors_num, 1]
            label: 为每个inside_anchors创建的label, 维度为: [inside_anchors_num, 1]
        """
        label = np.empty((len(inside_anchors)), dtype=np.int)
        # 先将label初始化为-1
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calculate_iou(inside_anchors, boxes)

        # 小于负样本的阈值就设置为负样本即将label设置为0
        label[max_ious < self.neg_iou_thresh] = 0
        # 大于正样本的阈值则设置为正样本即将label设置为1
        label[max_ious >= self.pos_iou_thresh] = 1
        # 将与真实框的iou重叠最大的anchors设置为正样本(分配每个真实框至少对应一个先验框)
        label[gt_argmax_ious] = 1

        # 随机采样, 正锚点和负锚点比例最高位1:1(忽略-1的锚点)
        pos_standard = int(self.pos_ratio * self.n_sample)
        pos_num = np.where(label == 1)[0]
        if len(pos_num) > pos_standard:
            # replace=False表示随机选择索引的时候不会重复
            disable_index = np.random.choice(pos_num, size=(len(pos_num) - pos_standard), replace=False)
            label[disable_index] = -1

        # 平衡正负样本，保持总数量为256
        neg_standard = self.n_sample - np.sum(label == 1)  # 非正样本的个数
        neg_num = np.where(label == 0)[0]
        if len(neg_num) > neg_standard:
            disable_index = np.random.choice(neg_num, size=(len(neg_num) - neg_standard), replace=False)
            label[disable_index] = -1
        return argmax_ious, label
