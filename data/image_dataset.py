from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from data.process_data import parse_xml
import numpy as np
from PIL import Image
import torch


class ImageDataset(Dataset):
    def __init__(self, xml_root_dir, img_root_dir, txt_root_dir, txt_file, transform=None):
        super(ImageDataset, self).__init__()

        self.xml_root_dir = xml_root_dir
        self.img_root_dir = img_root_dir
        self.txt_root_dir = txt_root_dir
        self.txt_file = txt_file
        if transform == None:
            self.transform = transforms.Compose([
                lambda x: Image.open(x).convert('RGB'),
                # 为了适配vgg16的输入
                transforms.Resize((int(224), int(224))),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])
        boxes, labels, images = self.load_txt(self.txt_file)
        self.boxes = boxes
        self.labels = labels
        self.images = images

    def load_txt(self, filename):
        """
        function description: 加载txt文件中的信息并放到numpy数组中, numpy可以直接在list中再次添加可变list

        :param filename: 文件名
        :return:
        """
        print('-------------the file name is ', filename)
        # boxes = list()
        boxes = []
        # labels = list()
        labels = []
        images = []
        print(os.path.join(self.txt_root_dir, filename))
        with open(os.path.join(self.txt_root_dir, filename), mode='r') as f:
            lines = f.readlines()
            index = 0
            for line in lines:
                # print(line.strip())
                box, label, image = self.load_xml(line.strip() + ".xml")
                # if index == 0:
                #     old_boxes = boxes
                #     old_labels = labels
                #     old_images = images
                # else:
                #     boxes = np.concatenate((old_boxes, boxes), axis=0)
                #     old_boxes = boxes
                #     labels = np.concatenate((old_labels, labels), axis=0)
                #     old_labels = labels
                #     images = np.concatenate((old_images, images))
                #     old_images = images
                boxes.append(box)
                labels.append(label)
                images.append(image)
                index += 1
        print('the length of boxes is ', len(boxes))
        print('the length of labels is ', len(labels))
        print('the length of images is ', len(images))
        return np.array(boxes), np.array(labels), np.array(images)

    def load_xml(self, filename):
        path = os.path.join(self.xml_root_dir, filename)
        if not os.path.exists(path):
            return
        boxes, labels = parse_xml(path)
        images = (self.img_root_dir + filename.replace('.xml', '.png'))
        return boxes, labels, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        box = self.boxes[index]
        label = self.labels[index]
        img_tensor = self.transform(self.images[index])
        return {
            "img_tensor": img_tensor,
            "img_classes": label,
            "img_gt_boxes": box
        }


if __name__ == '__main__':
    dataset = ImageDataset('../kitti/Annotations/', '../kitti/JPEGImages/data_object_image_2/training/image_2/',
                           '../kitti/ImageSets/Main/', 'test.txt')
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for index, sample in enumerate(loader):
        # sample是一个三个元素的tuple
        # 第一个元素是图片数据tensor，（1, 3, W, H）
        # 第二个元素是图片标记正负anchor样本的tensor,(1, #anchors)
        # 第三个元素是anchor的loc的tensor，其中非正样本的loc=0,(1, #anchors, 4)
        print(sample['img_tensor'].shape)
        print(sample['img_classes'][0])
        print(sample['img_gt_boxes'][0])
