import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import time
import os
from nets.faster_rcnn import FasterRCNN
from data.image_dataset import ImageDataset
from configs.config import epochs, lr, weight_decay


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.optimizer = self.get_optimizer(lr=lr, weight_decay=weight_decay)

    def forward(self, x, gt_boxes, labels):
        # start = time.time()
        losses = self.faster_rcnn(x, gt_boxes, labels)
        # end = time.time()
        # print("FasterRcnn forward 时间消耗: %s seconds" % (end - start))
        return losses

    def train_step(self, x, gt_boxes, labels):
        self.optimizer.zero_grad()
        losses = self.forward(x, gt_boxes, labels)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses.total_loss.item()

    def scale_lr(self, decay=0.1):
        """
        function description: 将优化其中的学习率缩小为1/10

        :param decay: 放缩倍率
        :return:
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    def save(self, save_optimizer=True, save_path=None, epoch=None,
             avg_train_loss=None, avg_test_loss=None):
        """
        function description: 保存model(包括optimizer以及其它信息), 返回存储的路径

        :param save_optimizer: 是否保存优化器
        :param save_path: 申明的存储路径
        :param epoch: 训练的epoch数
        :param avg_train_loss: 训练的平均loss
        :param avg_test_loss: 测试的平均loss
        :return:
            save_path: 保存的路径
        """
        save_dict = dict()
        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['epoch'] = epoch
        save_dict['train_loss'] = avg_train_loss
        save_dict['test_loss'] = avg_test_loss

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % self.optimizer.param_groups[0]["lr"]
            save_path += '-epoch-%d' % (epoch)
            save_path += '-trainloss-%.3f' % (avg_train_loss)
            save_path += 'testloss-%.3f' % (avg_test_loss)

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True):
        """
        function description: 读取model的state_dict, 返回对象自身

        :param path: 读取model的路径
        :param load_optimizer: 是否加载优化器
        :return:
        """
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def get_optimizer(self, lr, weight_decay):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer


if __name__ == '__main__':
    device = torch.device('cuda')
    path = 'pre_model_weights/vgg16-397923af.pth'
    faster_rcnn = FasterRCNN(path).to(device)
    trainer = FasterRCNNTrainer(faster_rcnn)
    # TODO 凭什么这里加的是相对faster_rcnn_trainer.py的文件
    dataset = ImageDataset('./kitti/Annotations/', './kitti/JPEGImages/data_object_image_2/training/image_2/',
                           './kitti/ImageSets/Main/', 'train.txt')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for epoch in range(epochs):
        for i, sample in enumerate(loader):
            x = sample['img_tensor'].to(device)
            gt_boxes = sample['img_gt_boxes'].to(device)
            # print(gt_boxes)
            labels = sample['img_classes'].to(device)
            losses = trainer.train_step(x, gt_boxes, labels)
            print('after ', i, 'step: loss=', losses)
