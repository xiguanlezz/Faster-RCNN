from torch import nn
import torch
import numpy as np


class RoIPooling2D(nn.Module):
    def __init__(self,
                 output_size,
                 spatial_scale,
                 return_indices=False):
        """
        function description: 将

        :param output_size:
        :param spatial_scale: 需要根据rois的坐标放缩到featuremap中的比例
        :param return_indices:
        """
        super(RoIPooling2D, self).__init__()

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.return_indices = return_indices
        # 将输入张量的维度变为[1, 1]
        self.adp_max_pool_2D = nn.AdaptiveMaxPool2d(output_size, return_indices)

    def forward(self, x, rois):
        """
        function description: 将roi的坐标映射到featuremap中的对应位置

        :param x: 预训练好的特征提取网络的输出即featuremap
        :param rois: 采样后的roi坐标
        :return:
        """
        rois_ = torch.from_numpy(rois).float()
        rois = rois_.mul(self.spatial_scale)
        rois = rois.long()

        num_rois = rois.size(0)
        output = []

        for i in range(num_rois):
            # roi维度为: [4]
            roi = rois[i]
            im = x[..., roi[0]:(roi[2] + 1), roi[1]:(roi[3] + 1)]
            try:
                output.append(self.adp_max_pool_2D(im))  # 元素维度 (1, channel, 7, 7)
            except RuntimeError:
                print("roi:", roi)
                print("raw roi:", rois[i])
                print("im:", im)
                print("outcome:", self.adp_max_pool_2D(im))

        output = torch.cat(output, 0)
        return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 50, 50)
    sample_rois = np.array([[0, 0, 17, 17], [0, 0, 31, 31], [0, 0, 64, 64], [0, 0, 128, 128]], dtype=float)
    roi_pooling_layer = RoIPooling2D((7, 7), 1. / 16)
    # print('before pooling:', sample_rois)
    output = roi_pooling_layer(x, sample_rois)
    # print('after pooling:', output)
