import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

# 数据预处理
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def main():
    # 加载本地数据集
    train_dir = r'D:\resnet\supervised_fault_classification\defect_supervised\glass-insulator\train'
    test_dir = r'D:\resnet\supervised_fault_classification\defect_supervised\glass-insulator\val'

    # 加载训练数据集
    train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=data_transform["train"])
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    # 加载测试集数据
    test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=data_transform["val"])
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("The size of Train_data is {}".format(train_data_size))
    print("The size of Test_data is {}".format(test_data_size))

    # 加载预训练的ResNet50模型
    resnet50 = models.resnet50(pretrained=True)
    num_ftrs = resnet50.fc.in_features

    # 冻结模型的参数
    for param in resnet50.parameters():
        param.requires_grad = False

    # 修改最后一层以适应新的类别数量
    resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, len(train_data.classes)),
                                nn.LogSoftmax(dim=1))

    # 将模型移动到指定设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 定义优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(resnet50.parameters(), lr=learning_rate)

    # 设置网络训练的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epochs = 1

    best_accuracy = 0.0

    for epoch in range(epochs):
        print("-------第{}轮训练开始-------".format(epoch + 1))

        # 训练步骤开始
        resnet50.train()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = resnet50(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {:.4f}".format(total_train_step, loss.item()))

        # 测试集
        resnet50.eval()
        total_test_loss = 0
        correct = 0

        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = resnet50(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total_test_step += 1

        average_test_loss = total_test_loss / len(test_dataloader)
        accuracy = correct / test_data_size

        print(
            "测试次数：{}, 测试Loss: {:.4f}, 准确率: {:.4f}%".format(total_test_step, average_test_loss, accuracy * 100))

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(resnet50.state_dict(), 'best_resnet50.pth')
            print(f"保存了更好的模型，准确率为 {accuracy * 100:.4f}%")

    print("训练完成")


if __name__ == '__main__':
    main()


