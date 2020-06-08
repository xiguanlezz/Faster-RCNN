from torch import nn
import torch
import torch.nn.functional as F
from nets.anchors_creator import generate_base_anchors, enumerate_shifted_anchor
from nets.proposal_creator import ProposalCreator
from utils.util import normal_init
from configs.config import anchors_ratios, anchors_scales, in_channels


class RPN(nn.Module):
    def __init__(self, in_channels, feature_stride=16):
        super(RPN, self).__init__()

        self.in_channels = in_channels
        self.anchor_scales = anchors_scales
        self.anchor_ratios = anchors_ratios
        # TODO 将该属性写到config中
        self.feature_stride = feature_stride

        # 可以把rpn传入; 如果是train阶段, 返回的roi数量是2000; 如果是test则是300
        self.proposal_layer = ProposalCreator(parent_model=self)

        self.base_anchors = generate_base_anchors(scales=self.anchor_scales, ratios=self.anchor_ratios)
        self.feature_stride = feature_stride

        # RPN的卷积层用来接收特征图, 输出512维的特征图
        self.RPN_conv = nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=1,
                                  bias=True)

        anchors_num = self.base_anchors.shape[0]
        # 2 x k(9) scores, 分类预测先验框内部是否包含物体
        self.RPN_cls_layer = nn.Conv2d(in_channels=512, out_channels=anchors_num * 2, kernel_size=1, stride=1,
                                       padding=0)
        # 4 x k(9) coordinates, 回归预测对先验框进行调整
        self.RPN_reg_layer = nn.Conv2d(in_channels=512, out_channels=anchors_num * 4, kernel_size=1, stride=1,
                                       padding=0)

        # paper中提到的用0均值高斯分布(标准差为0.01)初始化1x1卷积的权重
        normal_init(self.RPN_conv, 0, 0.01)
        normal_init(self.RPN_cls_layer, 0, 0.01)
        normal_init(self.RPN_reg_layer, 0, 0.01)

    @staticmethod
    def reshape(x, width):
        # input_size = x.size()
        # x = x.view(input_size[0], int(d), int(float(input_size[1] * input_size[2]) / float(d)), input_size[3])
        height = float(x.size(1) * x.size(1)) / width
        x = x.view(x.size(0), int(width), int(height), x.size(3))
        return x

    def forward(self, base_feature_map, img_size):
        n, _, w, h = base_feature_map.shape

        # TODO RPN Header
        # 前向传播的时候计算移动的anchors
        anchors = enumerate_shifted_anchor(self.base_anchors, base_size=self.feature_stride, width=w, height=h)

        h = F.relu(self.RPN_conv(base_feature_map), inplace=True)  # inplace=True表示原地操作, 节省内存

        # 回归预测, 输出维度: [n=1, anchors_number即KxP, 4], 其中第三个维度的四个点分别代表左上角和右下角的点的坐标
        rpn_locs = self.RPN_reg_layer(h)  # [1, 4*k, w, h] -> [1, w, h, 4*k]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 分类预测, 输出维度: [n=1, anchors_number即KxP, 2], 其中第三个维度为1表示检测到了object, 0表示未检测到object
        rpn_scores = self.RPN_cls_layer(h)  # [1, 2*k, w, h] -> [1, w, h, 2*k]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        print('rpn_locs: ', rpn_locs.shape)
        print('rpn_scores: ', rpn_scores.shape)

        # 根据rpn回归的结果对anchors微调之后, 还需要提供roi给fastrcnn部分
        rois = self.proposal_layer(rpn_locs[0].detach().cpu().numpy(),
                                   rpn_scores[0].detach().cpu().numpy(),
                                   anchors,
                                   img_size)

        return rpn_locs, rpn_scores, anchors, rois


if __name__ == '__main__':
    net = RPN(16)
    x = net(torch.ones((1, 16, 38, 38)), 19)
