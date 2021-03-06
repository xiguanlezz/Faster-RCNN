from faster_rcnn_trainer import FasterRCNNTrainer
from nets.faster_rcnn import FasterRCNN
from data.image_dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time
from configs.config import xml_root_dir, img_root_dir, txt_root_dir, device_name, pre_model_weights_path, epochs


@torch.no_grad()
def evaluate_test_data(testLoader, trainer):
    """
        这里我并没有使用VOC2007规定metric来评估,而只是从test dataset的loss来评估
    """
    test_loss = 0.0
    counter = 0
    device = torch.device(device_name)
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
    # 设置随机种子
    seed = 12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # python的三目运算符
    device = torch.device(device_name)
    path = pre_model_weights_path + 'vgg16-397923af.pth'
    faster_rcnn = FasterRCNN(path).to(device)
    trainer = FasterRCNNTrainer(faster_rcnn)
    # 训练集
    trainvalset = ImageDataset(xml_root_dir=xml_root_dir, img_root_dir=img_root_dir + 'training/',
                               txt_root_dir=txt_root_dir, txt_file='trainval.txt')
    trainvalLoader = DataLoader(trainvalset, batch_size=1, shuffle=True, num_workers=0)
    # 测试集
    testset = ImageDataset(xml_root_dir=xml_root_dir, img_root_dir=img_root_dir + 'training/',
                           txt_root_dir=txt_root_dir, txt_file='train_test.txt')
    testLoader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    trainer = trainer.to(device)

    already_trained_epoch = 0
    if already_trained_epoch != 0:
        file_name = "fasterrcnn_lr=0.005-epoch-2-trainloss-0.768testloss-0.758"
        load_path = 'checkpoints/' + file_name
        trainer.load(load_path)
        print("trained model loaded")
        print("loaded model lr: ", trainer.optimizer.param_groups[0]["lr"])  # 导入模型的学习率

    # 调节学习率
    change_lr = False
    if change_lr:
        scale = 0.1  # new_lr = lr* scale
        trainer.scale_lr(scale)


    for epoch in range(epochs):
        start = time.time()
        epoch_loss = 0.0

        for i, sample in tqdm(enumerate(trainvalLoader)):
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
