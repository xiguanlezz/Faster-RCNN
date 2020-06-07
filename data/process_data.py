from lxml import etree as ET
import glob
import cv2
import random


def write_xml(filename, saveimg, typename, boxes, xmlpath):
    """
    function description: 将txt的标注文件转为xml

    :param filename: 图片名
    :param saveimg: opencv读取图片
    :param typename: 类名
    :param boxes: 左上角和右下角坐标
    :param xmlpath: 保存的xml文件名
    :return:
    """
    # 根节点
    root = ET.Element("annotation")

    # folder节点
    folder_node = ET.SubElement(root, 'folder')
    folder_node.text = 'kitti'

    # filename节点
    filename_node = ET.SubElement(root, 'filename')
    filename_node.text = filename

    # source节点
    source_node = ET.SubElement(root, 'source')
    database_node = ET.SubElement(source_node, 'database')
    database_node.text = 'kitti Database'
    annotation_node = ET.SubElement(source_node, 'annotation')
    annotation_node.text = 'kitti'
    image_node = ET.SubElement(source_node, 'image')
    image_node.text = 'flickr'
    flickrid_node = ET.SubElement(source_node, 'flickrid')
    flickrid_node.text = '-1'

    # owner节点
    owner_node = ET.SubElement(root, 'owner')
    flickrid_node = ET.SubElement(owner_node, 'flickrid')
    flickrid_node.text = 'muke'
    name_node = ET.SubElement(owner_node, 'name')
    name_node.text = 'muke'

    # size节点
    size_node = ET.SubElement(root, 'size')
    width_node = ET.SubElement(size_node, 'width')
    width_node.text = str(saveimg.shape[1])
    height_node = ET.SubElement(size_node, 'height')
    height_node.text = str(saveimg.shape[0])
    depth_node = ET.SubElement(size_node, 'depth')
    depth_node.text = str(saveimg.shape[2])

    # segmented节点(用于图像分割)
    segmented_node = ET.SubElement(root, 'segmented')
    segmented_node.text = '0'

    # object节点(循环添加节点)
    for i in range(len(typename)):
        object_node = ET.SubElement(root, 'object')
        name_node = ET.SubElement(object_node, 'name')
        name_node.text = typename[i]
        pose_node = ET.SubElement(object_node, 'pose')
        pose_node.text = 'Unspecified'
        # 是否截断
        truncated_node = ET.SubElement(object_node, 'truncated')
        truncated_node.text = '1'
        difficult_node = ET.SubElement(object_node, 'difficult')
        difficult_node.text = '0'
        bndbox_node = ET.SubElement(object_node, 'bndbox')
        xmin_node = ET.SubElement(bndbox_node, 'xmin')
        xmin_node.text = str(boxes[i][0])
        ymin_node = ET.SubElement(bndbox_node, 'ymin')
        ymin_node.text = str(boxes[i][1])
        xmax_node = ET.SubElement(bndbox_node, 'xmax')
        xmax_node.text = str(boxes[i][2])
        ymax_node = ET.SubElement(bndbox_node, 'ymax')
        ymax_node.text = str(boxes[i][3])

    tree = ET.ElementTree(root)
    tree.write(xmlpath, pretty_print=True)


def parse_xml(filename):
    xml_file = ET.parse(filename)
    boxes = list()
    labels = list()
    for obj in xml_file.findall('object'):
        bndbox = obj.find('bndbox')
        boxes.append([int(bndbox.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
        labels.append(obj.find('name').text.lower().strip())
    return boxes, labels


def split_dataset():
    """
    function description: 将数据集切分为训练集, 验证集以及测试集, 并且写入相应的xml作为标注

    :return:
    """
    trainval = open('../kitti/ImageSets/Main/trainval.txt', 'w')
    train = open('../kitti/ImageSets/Main/train.txt', 'w')
    val = open('../kitti/ImageSets/Main/val.txt', 'w')
    test = open('../kitti/ImageSets/Main/test.txt', 'w')

    list_anno_files = glob.glob('../kitti/training/label_2/*')
    random.shuffle(list_anno_files)
    print(list_anno_files)
    index = 0
    for file_path in list_anno_files:
        with open(file_path) as file:
            boxes = []
            typename = []

            anno_infos = file.readlines()
            for anno_item in anno_infos:
                anno_new_infos = anno_item.split(" ")
                # 去掉杂项和不关心这俩类别
                if anno_new_infos[0] == "Misc" or anno_new_infos[0] == "DontCare":
                    continue
                else:
                    box = (int(float(anno_new_infos[4])), int(float(anno_new_infos[5])),
                           int(float(anno_new_infos[6])), int(float(anno_new_infos[7])))
                    boxes.append(box)
                    typename.append(anno_new_infos[0])

            filename = file_path.split("\\")[-1].replace("txt", "png")
            xmlpath = '../kitti/Annotations/' + filename.replace("png", "xml")
            imgpath = '../kitti/JPEGImages/data_object_image_2/training/image_2/' + filename
            saveimg = cv2.imread(imgpath)
            # print("xmlpath: ", xmlpath)
            write_xml(filename, saveimg, typename, boxes, xmlpath)

            index += 1
            if index < len(list_anno_files) * 0.7:
                train.write(filename.replace(".png", "\n"))
            elif index < len(list_anno_files) * 0.9:
                val.write(filename.replace(".png", "\n"))
            else:
                test.write(filename.replace(".png", "\n"))

    train.close()
    val.close()
    test.close()


if __name__ == '__main__':
    split_dataset()
