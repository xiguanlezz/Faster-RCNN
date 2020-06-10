from faster_rcnn_trainer import FasterRCNNTrainer
from nets.faster_rcnn import FasterRCNN
from data.image_dataset import ImageDataset
import torch
from torch.utils.data import DataLoader
from utils.draw_tool import draw_predict
from tqdm import tqdm

path = './pre_model_weights/vgg16-397923af.pth'

dataset = ImageDataset('./kitti/Annotations/', './kitti/JPEGImages/data_object_image_2/training/image_2/',
                       './kitti/ImageSets/Main/', 'test.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device('cuda')
faster_rcnn = FasterRCNN(path)
trainer = FasterRCNNTrainer(faster_rcnn)

trainer = trainer.to(device)

already_trained = True

if already_trained == True:
    # already_trained_epoch = 4
    load_path = 'checkpoints/' + 'fasterrcnn_06101531-epoch-8-trainloss-0.192testloss-0.816'
    trainer.load(load_path)

dir_path = './kitti/JPEGImages/data_object_image_2/testing/image_2/'
for i, sample in tqdm(enumerate(dataloader)):
    x = sample["img_tensor"].to(device)
    gt_boxes = sample["img_gt_boxes"].to(device)
    labels = sample["img_classes"].to(device)
    img_name = sample["img_name"][0]
    final_boxes, labels, scores = trainer.faster_rcnn.predict(x)
    # print(labels)
    draw_predict(dir_path, img_name, final_boxes, labels, scores)
