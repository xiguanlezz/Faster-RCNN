import numpy as np
import glob
import random
from lxml import etree as ET
import os
from PIL import Image
import csv

classes = (
    'person',
    'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
    'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
    'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
)


def split_dataset():
    """
    function description: 将数据集切分为训练集, 验证集以及测试集, 并写入train.txt  val.txt  trainval.txt

    """
    trainval = open('../VOC2007/ImageSets/Main/trainval.txt', 'w')
    train = open('../VOC2007/ImageSets/Main/train.txt', 'w')
    val = open('../VOC2007/ImageSets/Main/val.txt', 'w')
    test = open('../VOC2007/ImageSets/Main/test.txt', 'w')

    list_anno_files = glob.glob('../VOC2007/Annotations/*')
    random.shuffle(list_anno_files)
    print(len(list_anno_files))
    index = 0
    for file_path in list_anno_files:
        with open(file_path) as file:
            filename = file.name.split("\\")[1]
            filename = filename.replace(".xml", "\n")

            if index > len(list_anno_files) * 0.9:
                test.write(filename)
            else:
                trainval.write(filename)
                if index > len(list_anno_files) * 0.7:
                    val.write(filename)
                else:
                    train.write(filename)
            index += 1
    train.close()
    val.close()
    test.close()


def change_side_length(side_length, ratio):
    return np.round(side_length * ratio)


def parse_xml(filename):
    xml_file = ET.parse(filename)
    boxes = list()
    labels = list()
    for obj in xml_file.findall('object'):
        bndbox = obj.find('bndbox')
        boxes.append([int(bndbox.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
        label = obj.find('name').text.lower().strip()
        # index在1-20内表示物体列别, 0表示背景
        labels.append(classes.index(label) + 1)
    return boxes, labels


def save_to_csv(img_names, sizes, labels, boxes, des_file_name):
    """
    function description: 将标注信息写入csv文件中

    :param img_names:
    :param sizes:
    :param labels:
    :param boxes:
    :param des_file_name:
    """
    with open(des_file_name, mode='w', newline='') as f:
        writer = csv.writer(f)

        for index in range(len(img_names)):
            img_name = img_names[index]
            size = sizes[index]
            label = labels[index]
            box = boxes[index].astype(int)
            new_b = "["
            for i in range(len(box)):
                new_b += "["
                for j in range(len(box[i])):
                    new_b += str(box[i][j])
                    if j != len(box[i]) - 1:
                        new_b += ','
                new_b += "]"
                if i != len(box) - 1:
                    new_b += ','
            new_b += "]"
            writer.writerow([img_name, size, label, new_b])


def reshape(img, box=None):
    """
    function description: 将图片最小边的长度缩放为600

    """
    width, height = img.size
    min_side_length = height if width >= height else width

    ratio = 600.0 / min_side_length

    if min_side_length == width:
        new_width = int(600)
        new_height = int(np.round(height * ratio))
    elif min_side_length == height:
        new_height = int(600)
        new_width = int(np.round(width * ratio))

    # 原地改变img尺寸
    new_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    if box != None:
        box = np.array(box)
        box = box * ratio
        box = np.round(box)
        return new_img, box
    else:
        return new_img


def main():
    # split_dataset()

    txt_rootdir = "../VOC2007/ImageSets/Main/"
    xml_rootdir = "../VOC2007/Annotations/"
    img_rootdir = "../VOC2007/JPEGImages/testing/"
    with open(os.path.join(txt_rootdir, "test.txt")) as f:
        # img_names = []
        # sizes = []
        # labels = []
        # boxes = []
        lines = f.readlines()
        print(len(lines))
        for line in lines:
            line = line.strip()
            # box, label = parse_xml(xml_rootdir + line + ".xml")
            img_name = line + ".jpg"
            img = Image.open(img_rootdir + img_name)
            img = reshape(img)
            img.save('../VOC2007/JPEGImages/resize_test/' + img_name)
            # 处理成[w, h, channels_num]
            # size = list(img.size)
            # size.append(3)

            # img_names.append(img_name)
            # sizes.append(size)
            # labels.append(label)
            # boxes.append(box)

        # save_to_csv(img_names, sizes, labels, boxes, "../data/test.csv")


if __name__ == '__main__':
    main()
