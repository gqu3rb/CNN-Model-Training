import numpy as np
import ast
from torchvision import datasets
from torchvision.transforms import functional as F

def load_parameters(file_path):
    """解析量化參數檔，讀出 float 權重"""
    params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('==='):
            name = lines[i].split()[1]  # e.g. 'conv1.weight'
            i += 1
            scale_zp = lines[i].split('#')[1]
            scale = float(scale_zp.split(',')[0].split('=')[1])
            zp    = int(scale_zp.split(',')[1].split('=')[1])
            i += 1
            arr_lines = []
            while i < len(lines) and not lines[i].startswith('==='):
                arr_lines.append(lines[i])
                i += 1
            w_int = np.array(ast.literal_eval(''.join(arr_lines)), dtype=np.int16)
            # 反量化成 float
            w_fp = (w_int.astype(np.float32) - zp) * scale
            params[name] = w_fp
        else:
            i += 1
    return params

def conv2d_fp(x, w, pad=1):
    C_out, C_in, K, _ = w.shape
    H, W = x.shape[1], x.shape[2]
    x_p = np.pad(x, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    out = np.zeros((C_out, H, W), dtype=np.float32)
    for oc in range(C_out):
        for ic in range(C_in):
            for i in range(H):
                for j in range(W):
                    out[oc, i, j] += np.sum(x_p[ic, i:i+K, j:j+K] * w[oc, ic])
    return out

def relu_fp(x):
    return np.maximum(0, x)

def maxpool_fp(x, kernel):
    C, H, W = x.shape
    outH, outW = H//kernel, W//kernel
    out = np.zeros((C, outH, outW), dtype=np.float32)
    for c in range(C):
        for i in range(outH):
            for j in range(outW):
                out[c, i, j] = np.max(x[c,
                                        i*kernel:(i+1)*kernel,
                                        j*kernel:(j+1)*kernel])
    return out

def preprocess_fp(img):
    img16 = F.center_crop(img, 16)
    arr = np.array(img16, dtype=np.float32) / 255.0
    # 二值化後當作 float 輸入
    return (arr > 0.5).astype(np.float32)

def main():
    # 載入參數並反量化
    params = load_parameters('mnist_int8_params.txt')
    w1 = params['conv1.weight']  # shape (4,1,3,3)
    w2 = params['conv2.weight']  # shape (5,4,3,3)
    w3 = params['fc.weight']     # shape (10,5)

    # 載入測試集
    test_ds = datasets.MNIST('./data', train=False, download=False, transform=None)
    correct = 0

    for img, label in test_ds:
        x = preprocess_fp(img)[np.newaxis, ...]  # (1,16,16)
        # 第一層
        y = conv2d_fp(x, w1, pad=1)
        y = relu_fp(y)
        y = maxpool_fp(y, 2)
        # 第二層
        y = conv2d_fp(y, w2, pad=1)
        y = relu_fp(y)
        y = maxpool_fp(y, 8)
        # Flatten + fc
        y = y.reshape(-1)  # (5,)
        logits = w3.dot(y)  # (10,)
        pred = np.argmax(logits)
        if pred == label:
            correct += 1

    acc = correct / len(test_ds)
    print(f'浮點驗證準確度: {acc:.3f}')

if __name__ == '__main__':
    main()

