# 类名
class_num = 7
KITTI_BBOX_LABEL_NAMES = (
    'pedestrian',
    'truck',
    'car',
    'cyclist',
    'van',
    'tram',
    'person_sitting'
)

epochs = 5
# 学习率
lr = 0.005
# 设置正则化的惩戒项的系数
weight_decay = 0.0005

# rpn网络部分的参数
in_channels = 512  # rpn网络接受数据的维度, 即经过特征提取网络后输出的维度, vgg16是512
mid_channels = 512  # rpn网络中3 x 3卷积之后输出的维度
feature_stride = 16  # featuremap中感受野的大小
# 生成先验框的一些参数
anchors_scales = [8, 16, 32]
anchors_ratios = [0.5, 1, 2]
