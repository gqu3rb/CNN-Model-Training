import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, QConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization.observer import PerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization.fake_quantize import FakeQuantize
import multiprocessing
import numpy as np
import copy

actFakeQuant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)

wgtFakeQuant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric
)

# --------------------------
# 1. 定義 CNN 模型（支援 QAT）
# --------------------------
class SmallQuantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2) # 輸出 8x8

        self.conv2 = nn.Conv2d(4, 5, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(8)

        self.fc  = nn.Linear(5, 10, bias=False)

        # 融合 Conv+ReLU
        torch.quantization.fuse_modules(self, ['conv1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'relu2'], inplace=True)

    def forward(self, x):
        x = self.quant(x)  # ➤ fake quantization 開始

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
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
    for images, labels in loader: # Related to: Pytorch_Tutorial_1.pdf, P.39
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

def binarize_input(x): # 圖片像素二質化
    return (x > 0.5).float() # e.g. x = torch.tensor([0.2, 0.7, 0.3, 0.9])
                             #      return tensor([0.0, 1.0, 0.0, 1.0])

# --------------------------
# 3. 主程序
# --------------------------
def main():
    while 1:
        multiprocessing.freeze_support()

        torch.backends.quantized.engine = 'fbgemm'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("使用裝置:", device)


        transform = transforms.Compose([
            transforms.CenterCrop(16),
            transforms.ToTensor(), # 將圖片的儲存格式從 numpy 轉成 tensor，並將所有元素轉換為 0 到 1 間的浮點數
            transforms.Lambda(binarize_input)
        ])


        train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=10, shuffle=True,  num_workers=4, pin_memory=True) # shuffle=True 將訓練數據的順序打亂，避免過擬合
        test_loader  = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=4, pin_memory=True)

        # 初始化模型 + QAT 設定
        model = SmallQuantCNN().to(device)
        # 使全模型每層的 zero point 皆為 0
        model.qconfig = QConfig(
            activation=actFakeQuant,
            weight=wgtFakeQuant
        )
        torch.quantization.prepare_qat(model, inplace=True)

        print(f"模型總參數數量：{sum(p.numel() for p in model.parameters()):,d}")

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_int8_acc = 0.0
        last_best_record = 0
        best_int8_model = None
        for epoch in range(1, 76):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)

            # 將當前訓練出的模型量化為 int8，並評估精準度
            qat_cpu = copy.deepcopy(model).cpu().eval()
            int8_model = torch.quantization.convert(qat_cpu, inplace=False)
            _, int8_acc = eval_model(int8_model, test_loader, criterion, torch.device('cpu'))
            print(f'Epoch {epoch:2d} | Train Acc: {tr_acc:.3f} | Int8 Acc: {int8_acc:.3f}')

            # 更新並保存最佳 Int8 模型
            if int8_acc > best_int8_acc:
                last_best_record = 0
                best_int8_acc = int8_acc
                best_int8_model = int8_model
                torch.save(int8_model.state_dict(), 'best_qat_int8.pth')
            else:
                last_best_record += 1

            if last_best_record == 5:
                print("連續 5 次沒刷新準度紀錄，結束訓練")
                break

        torch.save(best_int8_model.state_dict(), f"mnist_int8_{best_int8_acc:.3f}.pth")
        # 將量化後的 best_qat 模型的權重和 scale/zero 匯出成 .txt
        with open(f"mnist_int8_params_{best_int8_acc:.3f}.txt", 'w') as f:
            for name, module in best_int8_model.named_modules():
                if isinstance(module, torch.nn.quantized.Conv2d) \
                or isinstance(module, torch.nn.quantized.Linear):
                    act_s, act_zp = module.scale, module.zero_point
                    w_q = module.weight()
                    # per-tensor
                    w_scales = w_q.q_scale()
                    w_zps    = w_q.q_zero_point()
                    w_int = w_q.int_repr().cpu().numpy()

                    f.write(f"=== {name} ===\n")
                    f.write(f"# act_scale={act_s}, act_zero_point={act_zp}\n")
                    f.write(f"# wt_scales={w_scales}, wt_zero_points={w_zps}\n")
                    f.write(np.array2string(
                        w_int,
                        separator=', ',
                        max_line_width=np.inf,
                        threshold=w_int.size+1
                    ) + '\n\n')
        print(f"量化後參數已匯出為 mnist_int8_params_{best_int8_acc:.3f}.txt (含 activation & weight qparams)")

        if best_int8_acc >= 0.99:
            break
        else:
            print("準確度小於 99%，重新訓練")
            print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n")

if __name__ == "__main__":
    main()
