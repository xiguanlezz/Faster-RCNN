import torch
import numpy as np


def normal_init(m, mean, stddev, truncated=False):
    """
    function description: 权重初始化函数

    :param m: 输入
    :param mean: 均值
    :param stddev: 标准差
    :param truncated: 是否截断, paper中使用矩阵奇异值分解加速的话就视为截断
    :return:
    """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def smooth_l1_loss(x, t, in_weight, sigma):
    """
    function description: 计算L1损失函数

    :param x: 输出的位置信息
    :param t: 标注的位置信息
    :param in_weight: 筛选矩阵, 非正样本的地方为0
    :param sigma:
    :return:
    """
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (abs_diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """
    function description: 仅对正样本进行loc_loss值的计算

    :param pred_loc: 输出的位置信息
    :param gt_loc: 标注的位置信息
    :param gt_label: 标注的类别
    :param sigma:
    :return:
    """
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # 用作筛选矩阵, 维度为[gt_label_num, 4]
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)

    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


def get_inside_index(anchors, img_width, img_height):
    """
    function description: 得到在图片内的anchors的第一个维度的索引

    :param anchors: RPN网络中输入的anchors, 维度[K*P, 4]
    :param img_width: 原始图片的宽度
    :param img_height: 原始图片的高度
    :return:
        inside_index: 返回在图片里面的anchors的第一个维度的索引
    """
    inside_index = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= img_width) &
        (anchors[:, 3] <= img_height)
    )[0]
    return inside_index  # 返回的维度是dim=0的索引


def calculate_iou(valid_anchors, boxes):
    """
    function description: 计算两个框框之间的iou(交集/并集)

    :param valid_anchors: 待计算的先验框, 维度为: [valid_anchors_num, 4]
    :param boxes: 图片中的真实标注框, 维度为: [boxes_num, 4]
    :return:
        ious: 每个inside_anchors和boxes的iou的二维张量, 维度为: [valid_anchors_num, boxes_num]
    """
    # 常规思路---对于两个矩形的左上角取最大值, 对于右下角取最小值, 再判断内部的矩形是否存在即可(不可取, 会报错)
    # ious = np.empty((len(valid_anchors), len(boxes)), dtype=np.float32)
    # ious.fill(0)
    # 命名规则: 左上角为1, 右下角为2
    # for i, point_i in enumerate(valid_anchors):
    #     xa1, ya1, xa2, ya2 = point_i
    #     anchor_area = (ya2 - ya1) * (xa2 - xa1)
    #     for j, point_j in enumerate(boxes):
    #         xb1, yb1, xb2, yb2 = point_j
    #         box_area = (yb2 - yb1) * (xb2 - xb1)
    #
    #         inter_x1 = max(xa1, xa2)
    #         inter_y1 = max(ya1, ya2)
    #         inter_x2 = min(xb1, xb2)
    #         inter_y2 = min(yb1, yb2)
    #         if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    #             overlap_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    #             iou = overlap_area * 1.0 / (anchor_area + box_area - overlap_area)
    #         else:
    #             iou = 0.
    #         ious[i][j] = iou

    # TODO 直接张量运算
    # 获得重叠面积最大化的左上角点的坐标信息, 返回的维度是[inside_anchors_num, boxes_num, 2]
    tl = np.maximum(valid_anchors[:, None, :2], boxes[:, :2])
    # 获得重叠面积最大化的右下角点的坐标信息, 返回的维度是[inside_anchors_num, boxes_num, 2]
    br = np.minimum(valid_anchors[:, None, 2:], boxes[:, 2:])

    # 计算重叠部分的面积, 返回的维度是[inside_anchors_num, boxes_num]
    area_overlap = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 计算inside_anchors的面积, 返回的维度是[inside_anchors_num]
    area_1 = np.prod(valid_anchors[:, 2:] - valid_anchors[:, :2], axis=1)
    # 计算boxes的面积, 返回的维度是[boxes_num]
    area_2 = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    # area_1[:, None]表示将数组扩张一个维度即维度变为[inside_anchors, 1]
    ious = area_overlap / (area_1[:, None] + area_2 - area_overlap)
    # 最后broadcast返回的维度是[inside_anchors_num, boxes_num]
    return ious


def loc2box(anchors, locs):
    """
    function description: 将所有的anchors根据通过rpn得到的locs值进行校正(因为先验框是有限的无法代表所有的情况, 需要进行调整)

    :param anchors: 先验框
    :param locs: rpn得到的locs
    :return:
        roi: 兴趣区域
    """
    anchors_width = anchors[:, 2] - anchors[:, 0]
    anchors_height = anchors[:, 3] - anchors[:, 1]
    anchors_center_x = anchors[:, 0] + 0.5 * anchors_width
    anchors_center_y = anchors[:, 1] + 0.5 * anchors_height

    tx = locs[:, 0]
    ty = locs[:, 1]
    tw = locs[:, 2]
    th = locs[:, 3]

    center_x = tx * anchors_width + anchors_center_x
    center_y = ty * anchors_height + anchors_center_y
    width = np.exp(tw) * anchors_width
    height = np.exp(th) * anchors_height

    # eps是一个很小的非负数, 使用eps将可能出现的零用eps来替换, 避免除数为0而报错
    roi = np.zeros(locs.shape, dtype=locs.dtype)
    roi[:, 0] = center_x - 0.5 * width  # xmin
    roi[:, 2] = center_x + 0.5 * width  # xmax
    roi[:, 1] = center_y - 0.5 * height  # ymin
    roi[:, 3] = center_y + 0.5 * height  # ymax
    return roi


def box2loc(valid_anchors, max_iou_gt_boxes):
    """
    function description: 计算每个anchor和其对应iou最大的gt_box之间的差距值

    :param valid_anchors: 带计算的valid_anchors
    :param max_iou_gt_boxes: 对应的gt_boxes
    :return:
    """
    width = valid_anchors[:, 2] - valid_anchors[:, 0]
    height = valid_anchors[:, 3] - valid_anchors[:, 1]
    center_x = valid_anchors[:, 0] + 0.5 * width
    center_y = valid_anchors[:, 1] + 0.5 * height

    base_width = max_iou_gt_boxes[:, 2] - max_iou_gt_boxes[:, 0]
    base_height = max_iou_gt_boxes[:, 3] - max_iou_gt_boxes[:, 1]
    base_center_x = max_iou_gt_boxes[:, 0] + 0.5 * base_width
    base_center_y = max_iou_gt_boxes[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps  # 最小值，防止除法溢出

    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dx = (base_center_x - center_x) / width
    dy = (base_center_y - center_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    # anchor_locs 2d array，(#index_inside,4)，4表示dx,dy,dw,dh
    anchor_locs = np.vstack((dx, dy, dw, dh)).transpose()

    return anchor_locs


def non_maximum_suppression(roi, thresh):
    """
    function description: 非极大值抑制算法, 将所有的rois放入一个数组中, 每次选出scores最高的roi并加入结果索引中,
                          分别和其他rois计算iou, 从数组中剔除iou超过阈值的rois, 一直重复这个步骤直到数组为空

    :param roi: 感兴趣的区域
    :param thresh: iou的阈值
    :return:
        roi_after_nms: 经过阈值的NMS后剩下rois
        keep: 经过阈值的NMS后剩下rois的索引
    """
    # 左上角点的坐标
    xmin = roi[:, 0]
    ymin = roi[:, 1]
    # 右下角点的坐标
    xmax = roi[:, 2]
    ymax = roi[:, 3]

    areas = (xmax - xmin) * (ymax - ymin)
    keep = []
    order = np.arange(roi.shape[0])
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # TODO 和计算iou有些许冗余
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])

        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)
        inter = width * height
        # 计算iou
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        idx = np.where(iou <= thresh)[0]  # 去掉和scores的iou大于阈值的roi
        order = order[1 + idx]  # 剔除score最大
    roi_after_nms = roi[keep]
    return roi_after_nms, keep
