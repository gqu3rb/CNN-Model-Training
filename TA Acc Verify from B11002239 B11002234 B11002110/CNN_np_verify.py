# 如果只要使用此驗證程式，只需依照需求更改有標註 "#!!" 的區塊即可
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
            i += 3 # 直接跳過參數 txt 檔中記錄 activation scale/zero 和 weight scale/zero 的行

            weight_arr_lines = []
            while i < len(lines) and not lines[i].startswith('==='): # 一直讀到下一個 '==='（或檔尾）
                weight_arr_lines.append(lines[i])
                i += 1
            arr = ast.literal_eval(''.join(weight_arr_lines))
            w_int = np.array(arr, dtype=np.int8)
            model_params[layer_name] = {
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
        self.w1 = w1.astype('int8')
        self.w2 = w2.astype('int8')
        self.w3 = w3.astype('int8')

        print(f"self.w1:\n{self.w1}\ntype:{type(self.w1)}\nshape:{self.w1.shape}")
        print(f"self.w2:\n{self.w2}\ntype:{type(self.w2)}\nshape:{self.w2.shape}")
        print(f"self.w3:\n{self.w3}\ntype:{type(self.w3)}\nshape:{self.w3.shape}")

        np.save('conv1 weight', w1)
        np.save('conv2 weight', w2)
        np.save('fc weight', w3)

    def forward(self, x):

        print(f"\nModel input x.shape = {x.shape}, x = \n{x}")

        # 第一層卷積
        # input channel: 1
        # output channel: 4
        # kernel size: 3x3
        # zero padding size: 1
        x = conv2d_int(x, self.w1, pad=1) # x 中的元素大小範圍: [9*-128, 9*127] = [-1152, 1143]
        print(f"\nAfter \"x = conv2d_int(x, self.w1, pad=1)\", x.shape = {x.shape}, x = \n{x}")

        x = relu_int(x) # x 中的元素大小範圍: [0, 1143]
        print(f"\nAfter \"x = relu_int(x)\", x.shape = {x.shape}, x = \n{x}")

        x = maxpool_int(x, 2)
        print(f"\nAfter \"x = maxpool_int(x, 2)\", x.shape = {x.shape}, x = \n{x}")

        # 第二層卷積
        # input channel: 4
        # output channel: 5
        # kernel size: 3x3
        # zero padding size: 1
        x = conv2d_int(x, self.w2, pad=1) # x 中的元素大小範圍: [1143*(9*4)*-128, 1143*(9*4)*127] = [-5266944, 5225796] = [-2^22.33, 2^22.32] -> 23+1 個 bit 存 (加上去的那個 1 是 sign bit)
        print(f"\nAfter \"x = conv2d_int(x, self.w2, pad=1)\", x.shape = {x.shape}, x = \n{x}")

        x = relu_int(x)
        print(f"\nAfter \"x = relu_int(x)\", x.shape = {x.shape}, x = \n{x}")

        x = x>>9 # 右移以確保能以有號 16bit 的形式存進 Data Memory。事實上這裡移 (24-16) 個 bit 就好，但為了讓整個流程位移的 bit 數都相等，所以才移到 9 個 bit
        print(f"\nAfter \"x = x>>9\", x.shape = {x.shape}, x = \n{x}")
        if x.max() > 32767 or x.min() < -32768: # 如果 x 中的元素大小超過 int16 範圍，暫停程式
            print(f"x = {x}")
            input('')

        x = maxpool_int(x, 8)
        print(f"\nAfter \"x = maxpool_int(x, 8)\", x.shape = {x.shape}, x = \n{x}")

        # 將 x reshape，以送至全連接層
        x = x.reshape(-1) # x 中的元素大小範圍: [-2^14, (2^14)-1] (因為剛剛右移了 9 個 bit)
        print(f"\nAfter \"x = x.reshape(-1)\", x.shape = {x.shape}, x = \n{x}")

        # 最後用來產生預測結果的全連接層
        # input channel: 5
        # output channel: 10
        x = np.dot(self.w3.astype(np.int32), x) # x 中的元素大小範圍: [((2^14)-1)*5*-128, (-2^14)*5*-128] = [-10485120, 10485760] = [-2^23.32, 2^23.32] -> 24+1 個 bit 存
        print(f"\nAfter \"x = np.dot(self.w3.astype(np.int32)\", x.shape = {x.shape}, x = \n{x}")

        x=x>>9 # 右移 (25-16) 個 bit 以確保能以有號 16bit 的形式存進 Data Memory
        print(f"\nAfter \"x = x>>9\", x.shape = {x.shape}, x = \n{x}")
        if x.max() > 32767 or x.min() < -32768: # 如果 x 中的元素大小超過 int16 範圍，暫停程式
            print(f"x = {x}")
            input('')

        pred = int(np.argmax(x))
        print(f"\npredict number: {pred}")

        return x

def preprocess_int(img): # 只保留圖片中央 16x16，並做二值化
    img16 = F.center_crop(img, 16)
    arr   = np.array(img16) / 255.0
    binar = (arr > 0.5).astype(np.int8)
    return binar

def main():
    # 載入 MNIST 測試集
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=None) # 由於 transform=None，因此 test_ds 的內容仍然是一張張的灰階 28x28 圖片和 label
    model = IntegerCNN('mnist_int8_params_0.780.txt') #!! 要測試的權重 txt 檔名稱
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

        #!! 如果要改成測整個資料集的準確度，就把這個 if 註解掉
        #if round_count == 1001:
        #    break

    acc = correct / (round_count-1)
    print(f'整數推論準確度: {acc:.3f}')

if __name__ == '__main__':
    main()
