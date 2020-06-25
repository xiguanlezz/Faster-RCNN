# 数据集部分配置(需要先提前改为VOC数据格式)
# 分类总数
class_num = 20
# 类别, label从1开始, 0表示背景
classes = (
    'person',
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
)
# xml文件根路径
xml_root_dir = 'D:/machineLearning/pycharmProjects/Faster-RCNN/VOC2007/Annotations/'
# img图片根路径, 后面需要自己根据是训练集还是测试集手动拼接上URI
img_root_dir = 'D:/machineLearning/pycharmProjects/Faster-RCNN/VOC2007/JPEGImages/'
# 训练的txt根路径
txt_root_dir = 'D:/machineLearning/pycharmProjects/Faster-RCNN/VOC2007/ImageSets/Main/'
pic_format = '.jpg'

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
