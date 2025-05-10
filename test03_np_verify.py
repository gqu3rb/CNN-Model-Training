import numpy as np
import ast
from torchvision import datasets
from torchvision.transforms import functional as F

#class SmallQuantCNN(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.quant   = QuantStub()
#        self.dequant = DeQuantStub()
#
#        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
#        self.relu1 = nn.ReLU(inplace=True)
#        self.pool1 = nn.MaxPool2d(2) # 輸出 8x8
#
#        self.conv2 = nn.Conv2d(4, 5, kernel_size=3, stride=1, padding=1, bias=False)
#        self.relu2 = nn.ReLU(inplace=True)
#        self.pool2 = nn.MaxPool2d(8)
#
#        self.fc  = nn.Linear(5, 10, bias=False)

def load_parameters(file_path):
    """解析量化參數檔，讀出 w_int"""
    params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('==='):
            # 取出層名稱
            name = lines[i].split()[1]  # e.g. 'conv1.weight'
            # 檔案格式：header → scale comment → array 開始
            i += 1
            # 解析 scale, zero_point 但整數推論時只用 zero_point 來還原 signed int
            scale_zp = lines[i].strip().split('#')[1]
            scale = float(scale_zp.split(',')[0].split('=')[1])
            zp    = int(scale_zp.split(',')[1].split('=')[1])
            i += 1
            # 讀權重矩陣
            arr_lines = []
            while i < len(lines) and not lines[i].startswith('==='): # 一直讀到下一個 '==='（或檔尾）
                arr_lines.append(lines[i])
                i += 1
            arr = ast.literal_eval(''.join(arr_lines))
            w_int = np.array(arr, dtype=np.int16)
            params[name] = {'w_int': w_int, 'zp': zp}
        else:
            i += 1
    return params

def conv2d_int(x, w, pad=1):
    """
    完全整數版卷積：
      x: (C_in, H, W), dtype=int8/int16
      w: (C_out, C_in, K, K), dtype=int16 (signed)
    回傳 int32 的累加結果（未乘任何 scale）
    """
    C_out, C_in, K, _ = w.shape
    H, W = x.shape[1], x.shape[2]
    x_p = np.pad(x, ((0,),(pad,),(pad,)), mode='constant').astype(np.int16)
    out = np.zeros((C_out, H, W), dtype=np.int32)
    for oc in range(C_out):
        for ic in range(C_in):
            for i in range(H):
                for j in range(W):
                    # 只做整數乘加
                    window = x_p[ic, i:i+K, j:j+K].astype(np.int32)
                    kern   = w[oc, ic].astype(np.int32)
                    out[oc, i, j] += np.sum(window * kern)
    return out

def relu_int(x):
    """整數版 ReLU"""
    # x 本身是 int32
    x[x < 0] = 0
    return x

def maxpool_int(x, kernel):
    """整數版最大池化"""
    C, H, W = x.shape
    outH, outW = H // kernel, W // kernel
    out = np.zeros((C, outH, outW), dtype=x.dtype)
    for c in range(C):
        for i in range(outH):
            for j in range(outW):
                patch = x[c,
                          i*kernel:(i+1)*kernel,
                          j*kernel:(j+1)*kernel]
                out[c, i, j] = np.max(patch)
    return out

class IntegerCNN:
    def __init__(self, param_file):
        p = load_parameters(param_file)
        # 將 w_int 轉成 signed weight: w_q = w_int - zp
        w1 = p['conv1.weight']['w_int']  - p['conv1.weight']['zp']
        w2 = p['conv2.weight']['w_int']  - p['conv2.weight']['zp']
        w3 = p['fc.weight']['w_int']     - p['fc.weight']['zp']
        # 轉成適合運算的 dtype
        self.w1 = w1.astype(np.int16)   # (4,1,3,3)
        self.w2 = w2.astype(np.int16)   # (5,4,3,3)
        self.w3 = w3.astype(np.int16)   # (10,5)

    def forward(self, x):
        # x: (1,16,16) 整數 0/1
        x = conv2d_int(x, self.w1, pad=1)   # → (4,16,16), int32
        x = relu_int(x)
        x = maxpool_int(x, 2)               # → (4,8,8)
        x = conv2d_int(x, self.w2, pad=1)   # → (5,8,8)
        x = relu_int(x)
        x = maxpool_int(x, 8)               # → (5,1,1)
        x = x.reshape(-1).astype(np.int32)  # → (5,)
        # 全連接： w3 (10,5) · x (5,) → (10,)
        return np.dot(self.w3.astype(np.int32), x)

def preprocess_int(img):
    """
    CenterCrop 16x16，ToTensor→float→二值化，最後轉 int8
    完全整數流程中，像素只保留 0 或 1
    """
    img16 = F.center_crop(img, 16)
    arr   = np.array(img16) / 255.0
    binar = (arr > 0.5).astype(np.int8)
    return binar

def main():
    # 載入 MNIST 測試集（請先下載好 ./data）
    test_ds = datasets.MNIST('./data', train=False,
                             download=False,
                             transform=None)
    model = IntegerCNN('mnist_int8_params.txt')
    correct = 0
    for img, label in test_ds:
        x = preprocess_int(img)[np.newaxis, ...]  # (1,16,16), int8
        logits = model.forward(x)
        pred = int(np.argmax(logits))
        if pred == label:
            correct += 1
    acc = correct / len(test_ds)
    print(f'整數推論準確度: {acc:.3f}')

if __name__ == '__main__':
    main()
