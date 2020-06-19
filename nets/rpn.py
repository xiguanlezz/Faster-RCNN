from torch import nn
import torch
import torch.nn.functional as F
from nets.anchors_creator import generate_base_anchors, enumerate_shifted_anchor
from nets.proposal_creator import ProposalCreator
from utils.util import normal_init
from configs.config import in_channels, mid_channels, feature_stride, anchors_scales, anchors_ratios


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.in_channels = in_channels  # 经过预训练好的特征提取网络输出的featuremap的通道数
        self.mid_channels = mid_channels  # rpn网络第一层3x3卷积层输出的维度
        self.feature_stride = feature_stride  # 可以理解为featuremap中感受野的大小(压缩的倍数)
        self.anchor_scales = anchors_scales  # 生成先验框的面积比例的开方
        self.anchor_ratios = anchors_ratios  # 生成先验框的宽高之比

        # 可以把rpn传入; 如果是train阶段, 返回的roi数量是2000; 如果是test则是300
        self.proposal_layer = ProposalCreator(parent_model=self)

        self.base_anchors = generate_base_anchors(scales=self.anchor_scales, ratios=self.anchor_ratios)
        self.feature_stride = feature_stride

        # RPN的卷积层用来接收特征图(预训练好的vgg16网络的输出)
        self.RPN_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=3, stride=1,
                                  padding=1)

        anchors_num = self.base_anchors.shape[0]
        # 2 x k(9) scores, 分类预测每一个网格点上每一个预测框内部是否包含了物体, 1表示包含了物体; 此处是1 x 1卷积, 只改变维度
        self.RPN_cls_layer = nn.Conv2d(in_channels=self.mid_channels, out_channels=anchors_num * 2, kernel_size=1,
                                       stride=1,
                                       padding=0)

        # 4 x k(9) coordinates, 回归预测每一个网格点上每一个先验框的变化情况; 此处是1 x 1卷积, 只改变维度
        self.RPN_reg_layer = nn.Conv2d(in_channels=self.mid_channels, out_channels=anchors_num * 4, kernel_size=1,
                                       stride=1,
                                       padding=0)

        # paper中提到的用0均值高斯分布(标准差为0.01)初始化1x1卷积的权重
        normal_init(self.RPN_conv, mean=0, stddev=0.01)
        normal_init(self.RPN_cls_layer, mean=0, stddev=0.01)
        normal_init(self.RPN_reg_layer, mean=0, stddev=0.01)

    def forward(self, base_feature_map, img_size):
        """
        function description: rpn网络的前向计算

        :param base_feature_map: 经过预训练好的特征提取网络后的输出, 维度为: [batch_size, 38, 38, 512]
        :param img_size: 原图的尺寸, 需要用这个对anchors进行才间再转化成rois
        :return:
            rpn_locs：rpn层回归预测每一个先验框的变化情况, 维度为:[n, w*h*k, 4]
            rpn_scores: rpn分类每一个预测框内部是否包含了物体, 维度为:[n, w*h*k, 2]
            anchors: featuremap中每个像素点生成k个先验框的集合, 维度为:[w*h*k ,4]
            rois: 通过rpn网络输出的locs来校正先验框anchors的位置并完成NMS之后的rois
        """
        n, _, w, h = base_feature_map.shape

        # 前向传播的时候计算移动的anchors
        anchors = enumerate_shifted_anchor(self.base_anchors, base_size=self.feature_stride, width=w, height=h)

        anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)

        x = F.relu(self.RPN_conv(base_feature_map), inplace=True)  # inplace=True表示原地操作, 节省内存

        # 回归预测, 其中第三个维度的四个数分别代表左上角和右下角的点的坐标
        rpn_locs = self.RPN_reg_layer(x)
        # [n, 4*k, w, h] -> [n, w, h, 4*k] -> [n, w*h*k, 4]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 分类预测, 其中第三个维度的第一个数表示类别标签(0为背景), 第二个数表示置信度
        rpn_scores = self.RPN_cls_layer(x)
        # [n, 2*k, w, h] -> [n, w, h, 2*k] -> [n, w*h*k, 2]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # TODO
        # [n, w, h, 2*k] -> [n, w, h, k, 2]
        rpn_scores = rpn_scores.view(n, w, h, anchor_num, 2)
        # [n, w, h, k, 2] -> [n, w*h*k, 2]
        rpn_scores = rpn_scores.view(n, -1, 2)

        # print('rpn_locs: ', rpn_locs.shape)
        # print('rpn_scores: ', rpn_scores.shape)

        # 根据rpn回归的结果对anchors微调以及裁剪之后转为rois, 同时提供rois给Fast-RCNN部分
        rois = self.proposal_layer(rpn_locs[0].detach().cpu().numpy(),
                                   rpn_scores[0].detach().cpu().numpy(),
                                   anchors,
                                   img_size)

        return rpn_locs, rpn_scores, anchors, rois

    @staticmethod
    def reshape(x, width):
        # input_size = x.size()
        # x = x.view(input_size[0], int(d), int(float(input_size[1] * input_size[2]) / float(d)), input_size[3])
        height = float(x.size(1) * x.size(1)) / width
        x = x.view(x.size(0), int(width), int(height), x.size(3))
        return x


if __name__ == '__main__':
    net = RPN()
    x = net(torch.ones((1, 512, 38, 38)), (224, 224))
