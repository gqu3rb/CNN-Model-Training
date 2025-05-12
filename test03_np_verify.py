import re # for re object
import numpy as np
import sys
import ast
from torchvision import datasets
from torchvision.transforms import functional as F
np.set_printoptions(threshold=sys.maxsize)

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

def load_parameters(file_path): # 讀取參數 txt 檔 (建議對照 mnist_int8_params.txt 內容來理解此段程式)
    model_params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('==='):
            layer_name = lines[i].split()[1]
            i += 1

            act_line = lines[i].lstrip('#').strip()
            parts = [p.strip() for p in act_line.split(',')]
            act_scale = float(parts[0].split('=')[1])
            act_zp    = int(parts[1].split('=')[1])
            i += 1

            wt_line = lines[i].lstrip('#').strip()
            m_scales = re.search(r'wt_scales=\[([^\]]*)\]', wt_line)
            m_zps    = re.search(r'wt_zero_points=\[([^\]]*)\]', wt_line)
            wt_scales = [float(x) for x in m_scales.group(1).split(',')]
            wt_zps    = [int(x)   for x in m_zps.group(1).split(',')]
            i += 1

            weight_arr_lines = []
            while i < len(lines) and not lines[i].startswith('==='): # 一直讀到下一個 '==='（或檔尾）
                weight_arr_lines.append(lines[i])
                i += 1
            arr = ast.literal_eval(''.join(weight_arr_lines))
            w_int = np.array(arr, dtype=np.int8)
            model_params[layer_name] = {
                'act_scale':   act_scale,
                'act_zero_point': act_zp,
                'wt_scales':   wt_scales,
                'wt_zero_points': wt_zps,
                'w_int':       w_int
            }
        else:
            i += 1
    return model_params

def conv2d_int(x, w, pad=1):
    out_channel_num, in_channel_num, K, _ = w.shape
    H, W = x.shape[1], x.shape[2]
    x_p = np.pad(x, ((0,),(pad,),(pad,)), mode='constant').astype(np.int32) # add padding to input
    # print(f"x_p:\n{x_p}\nshape: {x_p.shape}")
    out = np.zeros((out_channel_num, H, W), dtype=np.int32)
    for oc in range(out_channel_num):
        for ic in range(in_channel_num):
            for i in range(H):
                for j in range(W):
                    window = x_p[ic, i:i+K, j:j+K].astype(np.int32)
                    kern   = w[oc, ic].astype(np.int32)
                    out[oc, i, j] += np.sum(window * kern)
    return out

def relu_int(x):
    x[x < 0] = 0
    return x

def maxpool_int(x, kernel):
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
        w1 = p['conv1']['w_int'] # assume act_zp == 0
        w2 = p['conv2']['w_int'] # assume act_zp == 0
        w3 = p['fc']['w_int']
        wt_scale1 = p['conv1']['wt_scales']
        wt_scale2 = p['conv2']['wt_scales']
        wt_scale3 = p['fc']['wt_scales']
        self.w1 = w1.astype('int8')
        self.w2 = w2.astype('int8')
        self.w3 = w3.astype('int8')
        self.wt_scale1 = wt_scale1
        self.wt_scale2 = wt_scale2
        self.wt_scale3 = wt_scale3

        #print(f"self.w1:\n{self.w1}\ntype:{type(self.w1)}\nshape:{self.w1.shape}")
        #print(f"self.w2:\n{self.w2}\ntype:{type(self.w2)}\nshape:{self.w2.shape}")
        #print(f"self.w3:\n{self.w3}\ntype:{type(self.w3)}\nshape:{self.w3.shape}")

    def forward(self, x):
        print(f"input:\n{x}")
        x = conv2d_int(x, self.w1, pad=1)
        x = relu_int(x)
        print(f"After conv1+relu:\n{x}")
        print(f"before mul scale1, x.shape: {x.shape}")
        x = x*np.array(self.wt_scale1).reshape(-1, 1, 1)
        print(f"After mul scale1, x.shape: {x.shape}")
        x = maxpool_int(x, 2)
        print(f"After maxpool_1:\n{x}")
        x = conv2d_int(x, self.w2, pad=1)
        x = relu_int(x)
        print(f"before mul scale2, x.shape: {x.shape}")
        x = x*np.array(self.wt_scale2).reshape(-1, 1, 1)
        print(f"After mul scale2, x.shape: {x.shape}")
        x = maxpool_int(x, 8)
        print(f"After maxpool_2:\n{x}")
        x = x.reshape(-1)
        #print(f"After x.reshape(-1), x.shape = {x.shape}, x.dtype = {x.dtype}")
        x = np.dot(self.w3.astype(np.int32), x)
        print(f"After fc:\n{x}")
        print(f"x.shape: {x.shape}")
        x = np.array(self.wt_scale3).reshape(1, -1)*x
        return x

def preprocess_int(img): # 只保留圖片中央 16x16，並做二值化
    img16 = F.center_crop(img, 16)
    arr   = np.array(img16) / 255.0
    binar = (arr > 0.5).astype(np.int8)
    return binar

def main():
    # 載入 MNIST 測試集（請先下載好 ./data）
    test_ds = datasets.MNIST('./data', train=False,
                             download=True,
                             transform=None) # 由於 transform=None，因此 test_ds 的內容仍然是一張張的灰階 28x28 圖片和 label
    model = IntegerCNN('mnist_int8_params_0.753.txt')
    correct = 0
    round_count = 1
    for img, label in test_ds:
        print(f"\n\n========== round {round_count} ==========\n\n")
        round_count += 1

        print(f"original image:\n{img}")
        x = preprocess_int(img)[np.newaxis, ...]  # (1,16,16)
        #print(x)

        logits = model.forward(x)

        pred = int(np.argmax(logits))
        print(f"predict: {pred}")
        print(f"label: {label}")
        if pred == label:
            correct += 1
            print("CORRECT")
        else:
            print("WRONG")

    acc = correct / len(test_ds)
    print(f'整數推論準確度: {acc:.3f}')

if __name__ == '__main__':
    main()
