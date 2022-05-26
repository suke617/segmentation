import torch
from u_net_model import UNet
# from data_prepare import DataTransform,VOCDataset
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

#使用するネット構造の指定、学習した重みのロード
net= torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
state_dict=torch.load("./u_net.pth",map_location={'cuda:0 ': 'cpu' })
net.load_state_dict(state_dict)
net.eval() 

# 1. 元画像の表示
img_original = Image.open('tomato18.jpg')   # [高さ][幅][色RGB]
img_width, img_height = img_original.size


class BaseTransform():

   
    def __init__(self, input_size, color_mean, color_std):
        self.base_transform = transforms.Compose([
            transforms.Resize(input_size),  # 短い辺の長さがresizeの大きさになる
            transforms.ToTensor(),  # Torchテンソルに変換
            transforms.Normalize(color_mean, color_std)  # 色情報の標準化
        ])

    def __call__(self, img):
        return self.base_transform(img)

# Dataset作成
# (RGB)の色の平均値と標準偏差
input_size=256
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform=BaseTransform(input_size=input_size,color_mean=color_mean,color_std=color_std)
img_transformed =transform(img_original)
x_img = img_transformed.unsqueeze(0) # ミニバッチ化：torch.Size([1, 3, 475, 475])

x=x_img.to(device)
net.to(device)
output=net(x).cpu()

y = output 
y = y[0].detach().numpy().copy() # y：torch.Size([1, 2, 475, 475])
y = np.argmax(y, axis=0)
print(y)
result = Image.fromarray(np.uint8(y))
result = result.resize((img_width, img_height), Image.NEAREST)

result = result.convert('RGB')

for x in range(img_width):
    for y in range(img_height):
        pixel = result.getpixel((x,y))
        #RGBのどれかが0だったら抜き出す
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            #黒色にする
            img_original .putpixel((x,y),(0,0,0))
        else:
            continue

plt.imshow(img_original)
plt.show()