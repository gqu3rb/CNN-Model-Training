import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing

# --------------------------
# 1. 定義 CNN 模型（支援 QAT）
# --------------------------
class SmallQuantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(12, 10)

        # 融合 Conv+ReLU
        torch.quantization.fuse_modules(self, ['conv1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'relu2'], inplace=True)

    def forward(self, x):
        x = self.quant(x)  # ➤ fake quantization 開始

        x = self.pool1(self.relu1(self.conv1(x)))  # (B,6,8,8)
        x = self.pool2(self.relu2(self.conv2(x)))  # (B,12,4,4)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.dequant(x)  # ➤ fake quantization 結束
        return x

# --------------------------
# 2. 訓練 / 驗證 函數
# --------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

# --------------------------
# 3. 主程序
# --------------------------
def binarize_input(x):
    return (x > 0.5).float()

def main():
    multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用裝置:", device)


    transform = transforms.Compose([
        transforms.CenterCrop(16),
        transforms.ToTensor(),
        transforms.Lambda(binarize_input)  # 替代 lambda
        # transforms.Normalize((0.5,), (0.5,))
    ])


    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型 + QAT 設定
    model = SmallQuantCNN().to(device)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    print(f"模型總參數數量：{sum(p.numel() for p in model.parameters()):,d}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, 11):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch:2d} | Train Acc: {tr_acc:.3f} | Val Acc: {val_acc:.3f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_qat.pth')
        if best_acc >= 0.80:
            print("已達 80% 準確度，提前停止訓練")
            break

    # 轉為 INT8 真正量化模型
    print("轉換為 INT8 量化模型...")
    model.cpu().eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), 'mnist_int8_2.pth')
    print("量化模型已儲存為 mnist_int8.pth")

    # 測試
    test_loss, test_acc = eval_model(quantized_model, test_loader, criterion, torch.device('cpu'))
    print(f'量化後測試準確度: {test_acc:.3f}')

if __name__ == "__main__":
    main()
