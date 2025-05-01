import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing

# --------------------------
# 1. 定義一個簡單的量化 CNN 模型
# --------------------------
class SmallQuantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 量化／反量化 stub，用於 QAT
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        # 兩層卷積 + 池化
        # Conv1: 輸入 1 通道 → 輸出 6 通道，kernel 3×3，padding=1 保持尺寸
        # 輸入 shape: (B,1,16,16)  （事先 crop 到 16×16）
        # 輸出 shape 經 conv1 後還是 (B,6,16,16)
        # 參數量 = 6×1×3×3 + 6 = 54 + 6 = 60
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # 池化 2×2 → spatial 大小: 16→8
        self.pool1 = nn.MaxPool2d(2)   # (B,6,16,16) → (B,6,8,8)

        # Conv2: 6 → 12 通道，同樣 3×3 + padding
        # 輸入 shape: (B,6,8,8)
        # 輸出 shape: (B,12,8,8)
        # 參數量 = 12×6×3×3 + 12 = 648 + 12 = 660
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # 池化 2×2 → spatial 大小: 8→4
        self.pool2 = nn.MaxPool2d(2)   # (B,12,8,8) → (B,12,4,4)

        # 全局平均池化：把 (B,12,4,4) 變成 (B,12,1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 最後一層線性：12 → 10
        # 輸入 features = 12*1*1 = 12
        self.fc  = nn.Linear(12, 10)

        # Fuse Conv+ReLU
        torch.quantization.fuse_modules(self, ['conv1','relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2','relu2'], inplace=True)

    def forward(self, x):
        # 假量化：把浮點數輸入轉到 fake-quant 範圍
        x = self.quant(x)

        # 卷積 + 池化
        x = self.pool1(self.relu1(self.conv1(x)))  # (B,6,8,8)
        x = self.pool2(self.relu2(self.conv2(x)))  # (B,12,4,4)

        # 全局平均池化
        x = self.gap(x)                            # (B,12,1,1)

        # 攤平成 (B,12)
        x = x.reshape(x.size(0), -1)

        # 分類頭
        x = self.fc(x)                             # (B,10)

        # 反量化：轉回浮點輸出
        return self.dequant(x)


# --------------------------
# 2. 訓練／驗證函式 （不變）
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
# 3. 主程序：放在 guard 裡面，支援 Windows spawn
# --------------------------
def main():
    multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用裝置:", device)

    # 資料前處理：先從 28×28 中心裁切到 16×16，再 ToTensor → Normalize
    transform = transforms.Compose([
        transforms.CenterCrop(16),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型與 QAT
    model = SmallQuantCNN().to(device)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    print(f"全參數量：{sum(p.numel() for p in model.parameters()):,d}")
    print(f"可訓練參數量：{sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 訓練迴圈
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

    # 量化並測試
    print("開始轉換到 int8 模型...")
    model.cpu().eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), 'mnist_int8.pth')
    print("量化模型已儲存為 mnist_int8.pth")

    quantized_model.eval()
    test_loss, test_acc = eval_model(quantized_model, test_loader, criterion, torch.device('cpu'))
    print(f'量化後測試準確度: {test_acc:.3f}')

if __name__ == "__main__":
    main()
