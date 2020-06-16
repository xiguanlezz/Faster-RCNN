from faster_rcnn_trainer import FasterRCNNTrainer
from nets.faster_rcnn import FasterRCNN
from data.image_dataset import ImageDataset
import torch
from torch.utils.data import DataLoader
from utils.draw_tool import draw_predict
from tqdm import tqdm
from configs.config import xml_root_dir, img_root_dir, txt_root_dir

path = './pre_model_weights/vgg16-397923af.pth'

dataset = ImageDataset(xml_root_dir=xml_root_dir, img_root_dir=img_root_dir + 'resize_test/',
                       txt_root_dir=txt_root_dir, txt_file='test.txt', isTest=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device('cuda')
faster_rcnn = FasterRCNN(path)
trainer = FasterRCNNTrainer(faster_rcnn)

trainer = trainer.to(device)

already_trained = True

if already_trained == True:
    # already_trained_epoch = 4
    load_path = 'checkpoints/' + 'fasterrcnn_0.005-epoch-1-trainloss-55953033047660160.000testloss-0.363'
    trainer.load(load_path)

dir_path = './VOC2007/JPEGImages/resize_test/'
for i, sample in tqdm(enumerate(dataloader)):
    x = sample["img_tensor"].to(device)
    # gt_boxes = sample["img_gt_boxes"].to(device)
    # labels = sample["img_classes"].to(device)
    img_name = sample["img_name"][0]
    final_boxes, labels, scores = trainer.faster_rcnn.predict(x)
    # print(labels)
    draw_predict(dir_path, img_name, final_boxes, labels, scores)
