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
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 兩層卷積 + 池化
        # Conv1: 輸入 1 通道 → 輸出 {2} 通道，kernel 3×3，padding=1 保持尺寸
        # 輸入 shape: (B,1,24,24)  （事先 crop 到 24×24）
        # 輸出 shape 經 conv1 後還是 (B,{2},24,24)
        # 參數量 = out_ch×in_ch×K×K + out_ch_bias = {2}×1×3×3 + {2}
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2)   # 24×24 → 12×12
        # 池化後 spatial 大小 floor(24/2)=12
        # 輸出 shape: (B,4,12,12)
        # 池化層沒有參數

        # Conv2: 6 → 12 通道，同樣 3×3 + padding
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)   # 12×12 → 6×6

        # 全局平均池化：把 8x6x6 變成 12x1x1
        self.gap   = nn.AdaptiveAvgPool2d(1)
        # 最後一層線性：12 → 10
        self.fc    = nn.Linear(12, 10)

        # 在 QAT 前將 Conv+ReLU、FC+ReLU fuse
        torch.quantization.fuse_modules(self, ['conv1','relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2','relu2'], inplace=True)

    def forward(self, x):
        # 假量化：把浮點數輸入轉到 fake-quant 範圍
        x = self.quant(x)

        # 卷積 + 池化
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.gap(x)

        # 攤平成 (B,12)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)             # [B,10]

        # 反量化：轉回浮點輸出
        return self.dequant(x)

# --------------------------
# 2. 訓練／驗證函式
# --------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()   # 切到訓練模式
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()       # 清除過去梯度
        outputs = model(images)     # forward
        loss = criterion(outputs, labels)
        loss.backward()             # backward
        optimizer.step()            # 更新參數

        total_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

def eval_model(model, loader, criterion, device):
    model.eval()    # 切到評估模式（BatchNorm/Dropout 等會自動關閉）
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

def count_all_parameters(m):
    return sum(p.numel() for p in m.parameters())

def count_trainable_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# --------------------------
# 3. 主程序：放在 guard 裡面，支援 Windows spawn
# --------------------------
def main():
    multiprocessing.freeze_support()  # Windows 上多進程必備

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用裝置:", device)

    # 資料前處理：先 Crop 24×24，再轉 Tensor、Normalize
    transform = transforms.Compose([
        transforms.CenterCrop(24),                # 從 28×28 中心位置裁切
        transforms.ToTensor(),                    # [0,255]→[0,1]
        transforms.Normalize((0.1307,), (0.3081,))# 常用 MNIST normalize
    ])

    # 下載 MNIST
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # DataLoader：batch 128，可自行調整 num_workers
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型、量化設定 (QAT)
    model = SmallQuantCNN().to(device)

    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    print(f"全參數量：{count_all_parameters(model):,d}")
    print(f"可訓練參數量：{count_trainable_parameters(model):,d}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # 4. 訓練迴圈 (QAT)
    # --------------------------
    best_acc = 0.0
    for epoch in range(1, 11):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, test_loader, criterion, device)
        print(f'Epoch {epoch:2d} | Train Acc: {tr_acc:.3f} | Val Acc: {val_acc:.3f}')

        # 儲存最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_qat.pth')

        # 如果已達到 0.80，則提前結束
        if best_acc >= 0.80:
            print("已達 80% 準確度，提前停止訓練")
            break

    # --------------------------
    # 5. 轉換到真實 int8 模型 & 最終測試
    # --------------------------
    print("開始轉換到 int8 模型...")
    model.cpu().eval()  # 量化轉換在 CPU 上進行
    quantized_model = torch.quantization.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), 'mnist_int8.pth')
    print("量化模型已儲存為 mnist_int8.pth")

    # 一定要留在 main() 裡，並且只用 CPU 評估
    quantized_model.eval()
    # 直接用 CPU，不要 .to(device)
    test_loss, test_acc = eval_model(quantized_model, test_loader, criterion, torch.device('cpu'))
    print(f'量化後測試準確度: {test_acc:.3f}')

if __name__ == "__main__":
    main()
