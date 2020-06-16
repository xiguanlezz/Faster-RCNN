from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from data.process_data import parse_xml, reshape
import numpy as np
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, xml_root_dir, img_root_dir, txt_root_dir, txt_file, isTest=False, transform=None):
        super(ImageDataset, self).__init__()

        self.xml_root_dir = xml_root_dir
        self.img_root_dir = img_root_dir
        self.txt_root_dir = txt_root_dir
        self.txt_file = txt_file
        self.isTest = isTest
        if transform == None:
            self.transform = transforms.Compose([
                # TODO BUG的根源... 为了适配vgg16的输入
                # transforms.Resize((int(224), int(224))),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])
        if self.isTest == False:
            boxes, labels, images = self.load_txt(self.txt_file)
            self.boxes = boxes
            self.labels = labels
            self.images = images
        elif self.isTest == True:
            self.images = self.load_txt(self.txt_file)

        id_list_files = os.path.join(txt_root_dir, txt_file)
        self.ids = [id_.strip() for id_ in open(id_list_files)]

    def load_txt(self, filename):
        """
        function description: 加载txt文件中的信息并放到numpy数组中, numpy可以直接在list中再次添加可变list

        :param filename: txt文件名
        """
        print('-------------the file name is ', filename)
        boxes = []
        labels = []
        images = []
        print(os.path.join(self.txt_root_dir, filename))
        with open(os.path.join(self.txt_root_dir, filename), mode='r') as f:
            lines = f.readlines()
            # index = 0
            for line in lines:
                line = line.strip()
                if self.isTest == False:
                    box, label, image = self.load_xml(line + ".xml")
                    boxes.append(box)
                    labels.append(label)
                    # index += 1
                elif self.isTest == True:
                    image = (line + ".jpg")
                    # image = line.replace("\n", ".jpg")
                images.append(image)

        if self.isTest == False:
            print('the length of boxes is ', len(boxes))
            print('the length of labels is ', len(labels))
            print('the length of images is ', len(images))
            return boxes, labels, images
        elif self.isTest == True:
            return images

    def load_xml(self, filename):
        """
        function description: 加载xml文件中需要的属性并将最小边缩放为600

        :param filename: xml文件名
        """
        path = os.path.join(self.xml_root_dir, filename)
        if not os.path.exists(path):
            return

        boxes, labels = parse_xml(path)
        img_name = filename.replace(".xml", ".jpg")
        images, boxes = reshape(Image.open(self.img_root_dir + img_name), boxes)
        return np.stack(boxes).astype(np.float32), \
               np.stack(labels).astype(np.int32), \
               images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.isTest == False:
            id = self.ids[index]
            box, label, image = self.load_xml('{0}.xml'.format(id))
            img_tensor = self.transform(image)
            return {
                "img_name": id + ".png",
                "img_tensor": img_tensor,
                "img_classes": label,
                "img_gt_boxes": box
            }
        elif self.isTest == True:
            img = Image.open(self.img_root_dir + self.images[index])
            img_tensor = self.transform(img)
            return {
                "img_name": self.images[index] + ".png",
                "img_tensor": img_tensor,
                # "img_classes": label,
                # "img_gt_boxes": box
            }


if __name__ == '__main__':
    dataset = ImageDataset(xml_root_dir='../kitti/Annotations/', img_root_dir='../kitti/JPEGImages/training/',
                           txt_root_dir='../kitti/ImageSets/Main/', txt_file='train.txt')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for index, sample in enumerate(loader):
        # sample是一个三个元素的tuple
        # 第一个元素是图片数据tensor，（1, 3, W, H）
        # 第二个元素是图片标记正负anchor样本的tensor,(1, #anchors)
        # 第三个元素是anchor的loc的tensor，其中非正样本的loc=0,(1, #anchors, 4)
        print(sample['img_tensor'].shape)
        print(sample['img_classes'][0])
        print(sample['img_gt_boxes'][0])
