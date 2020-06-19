from torchvision import models
from torch import nn
import torch


def decom_VGG16(path):
    model = load_pretrained_vgg16(path)
    # print(model)
    # 拿出vgg16模型的前30层来进行特征提取
    features = list(model.features)[:30]
    features = nn.Sequential(*features)

    # 获取vgg16的分类的那些层
    classifier = list(model.classifier)
    # 除去Dropout的相关层
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 前10层的参数不进行更新
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return features, classifier


def load_pretrained_vgg16(path):
    vgg16 = models.vgg16()
    vgg16.load_state_dict(torch.load(path))
    return vgg16
    # return models.vgg16(pretrained=True)


if __name__ == '__main__':
    path = '../vgg16-397923af.pth'
    x = torch.rand((1, 3, 700, 850))  # 第一张图片
    # model = torch.load(path)
    # vgg16 = models.vgg16()
    # vgg16.load_state_dict(torch.load(path))
    features, classifier = decom_VGG16(path)
    print(features)
    print(classifier)
    print(features(x).shape)
