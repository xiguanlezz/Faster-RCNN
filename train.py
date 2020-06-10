from faster_rcnn_trainer import FasterRCNNTrainer
from nets.faster_rcnn import FasterRCNN
from data.image_dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time


@torch.no_grad()
def evaluate_test_data(testLoader, trainer):
    """
        这里我并没有使用VOC2007规定metric来评估,而只是从test dataset的loss来评估
    """
    test_loss = 0.0
    counter = 0
    device = torch.device('cuda')
    for i, sample in tqdm(enumerate(testLoader)):
        x = sample["img_tensor"].to(device)
        gt_boxes = sample["img_gt_boxes"].to(device)
        labels = sample["img_classes"].to(device)
        loss = trainer(x, gt_boxes, labels).total_loss.item()
        test_loss += loss
        counter += 1
    avg_test_loss = test_loss * 1.0 / counter
    return avg_test_loss


def main():
    device = torch.device('cuda')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'pre_model_weights/vgg16-397923af.pth'
    faster_rcnn = FasterRCNN(path).to(device)
    trainer = FasterRCNNTrainer(faster_rcnn)
    # 训练集
    trainvalset = ImageDataset('./kitti/Annotations/', './kitti/JPEGImages/data_object_image_2/training/image_2/',
                               './kitti/ImageSets/Main/', 'trainval.txt')
    trainvalLoader = DataLoader(trainvalset, batch_size=1, shuffle=True, num_workers=0)
    # 测试集
    testset = ImageDataset('./kitti/Annotations/', './kitti/JPEGImages/data_object_image_2/training/image_2/',
                           './kitti/ImageSets/Main/', 'test.txt')
    testLoader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    trainer = trainer.to(device)

    already_trained_epoch = 0
    if already_trained_epoch != 0:
        file_name = "fasterrcnn_06101508-epoch-6-trainloss-0.204testloss-0.820"
        load_path = 'checkpoints/' + file_name
        trainer.load(load_path)
        print("trained model loaded")
        print("loaded model lr: ", trainer.optimizer.param_groups[0]["lr"])  # 导入模型的学习率

    # 调节学习率
    change_lr = True
    if change_lr:
        scale = 0.4  # new_lr = lr* scale
        trainer.scale_lr(scale)
        print("learning rate be smaller")

    total_epochs = 20

    for epoch in range(total_epochs):
        start = time.time()
        epoch_loss = 0.0

        for i, sample in tqdm(enumerate(trainvalLoader)):
            # print(sample["img_tensor"].size)
            x = sample["img_tensor"].to(device)
            gt_boxes = sample["img_gt_boxes"].to(device)
            labels = sample["img_classes"].to(device)

            loss = trainer.train_step(x, gt_boxes, labels)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss * 1.0 / len(trainvalLoader)  # epoch平均loss

        avg_test_loss = evaluate_test_data(testLoader, trainer)
        trainer.save(save_optimizer=True,
                     epoch=epoch + 1,
                     avg_train_loss=avg_epoch_loss,
                     avg_test_loss=avg_test_loss)
        end = time.time()
        print('after an epoch, time consumes ', end - start)


if __name__ == '__main__':
    main()
