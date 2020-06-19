from torch import nn
from nets.roi_pooling_2d import RoIPooling2D
from nets.vgg16 import decom_VGG16
from utils.util import normal_init


class FastRCNN(nn.Module):
    def __init__(self,
                 n_class,
                 roi_size,
                 spatial_scale,
                 classifier):
        """
        function description:
            将rpn网络提供的roi"投射"到vgg16的featuremap上, 进行相应的切割并maxpooling(RoI maxpooling),
            再将其展开从2d变为1d,投入两个fc层,然后再分别带入两个分支fc层，作为cls和reg的输出

        :param n_class: 分类的总数
        :param roi_size: RoIPooling2D之后的维度
        :param spatial_scale: roi(rpn推荐的区域-原图上的区域)投射在feature map后需要缩小的比例, 这个个人感觉应该对应感受野大小
        :param classifier: 从vgg16提取的两层fc(Relu激活)
        """
        super(FastRCNN, self).__init__()

        self.classifier = classifier
        self.cls_layer = nn.Linear(4096, n_class)
        self.reg_layer = nn.Linear(4096, n_class * 4)
        normal_init(self.cls_layer, mean=0, stddev=0.001)
        normal_init(self.reg_layer, mean=0, stddev=0.01)
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, sample_rois):
        """
        function decsription: Fast-RCNN的前向传播

        :param x: 预训练好的特征提取网络的输出即featuremap
        :param sample_rois: 经过NMS后的rois
        :return:
            roi_locs: roi的回归损失
            roi_scores: roi的分类损失
        """
        # Roi pooling
        pool = self.roi(x, sample_rois)
        # [n, 7, 7, 512] -> [n, 25088]  展平喂入全连接层
        pool = pool.view(pool.size(0), -1)
        # vgg16最后的两层全连接层
        fc7 = self.classifier(pool)
        roi_scores = self.cls_layer(fc7)
        roi_locs = self.reg_layer(fc7)
        return roi_locs, roi_scores


if __name__ == '__main__':
    import numpy as np
    import torch

    path = 'pretrained_model/checkpoints/vgg16-397923af.pth'
    _, classifier = decom_VGG16(path)
    fast_rcnn = FastRCNN(21, 7, 1. / 16, classifier)
    x = torch.randn(1, 512, 50, 50)
    sample_rois = np.array([[0, 0, 17, 17],
                            [0, 0, 31, 31, ],
                            [0, 0, 64, 64],
                            [0, 0, 128, 128]], dtype="float32")
    roi_cls_locs, roi_scores = fast_rcnn(x, sample_rois)
