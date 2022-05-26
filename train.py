import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import glob
from tqdm import tqdm 

"""自作ライブラリ"""
from u_net_model import UNet,IoU , DiceLoss
from data_prepare import DataTransform,VOCDataset

#ファイルパス持ってくる
def make_datapath_list(Phase="train"):
    if Phase=="train":
        rootpath1='data/images'
        rootpath2='data/ano_datas'
        input=glob.glob(rootpath1+'/*.jpg')
        annotation=glob.glob(rootpath2+'/*.png')
    else :
        rootpath1='data/image'
        rootpath2='data/ano_data'
        input=glob.glob(rootpath1+'/*.jpg')
        annotation=glob.glob(rootpath2+'/*.png')
    return input,annotation

input_train,annotation_train=make_datapath_list(Phase="train")
input_val,annotation_val=make_datapath_list(Phase="val")


# Dataset作成
# (RGB)の色の平均値と標準偏差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)


train_dataset = VOCDataset(input_train, annotation_train, phase="train", transform=DataTransform(
    input_size=256, color_mean=color_mean, color_std=color_std))

val_dataset = VOCDataset(input_val, annotation_val, phase="val", transform=DataTransform(
    input_size=256, color_mean=color_mean, color_std=color_std))


# DataLoader作成
batch_size = 8

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict , scheduler, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # multiple minibatch
    batch_multiplier = 3

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss=[]  # epochの損失和
        epoch_val_loss=[]  # epochの損失和

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')


        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                optimizer.step()  # 最適化schedulerの更新
                optimizer.zero_grad()
                # print('（train）')

            else:
                if((epoch+1) % 5 == 0):
                    net.eval()   # モデルを検証モードに
                    # print('-------------')
                    # print('（val）')
                else:
                    # 検証は5回に1回だけ行う
                    continue

            pbar=tqdm(dataloaders_dict[phase])
            count = 0  # multiple minibatc

            # データローダーからminibatchずつ取り出すループ
            for imges, anno_class_imges in pbar :
                # for imges, anno_class_imges in dataloaders_dict[phase]:
                # ミニバッチがサイズが1だと、バッチノーマライゼーションでエラーになるのでさける

                # GPUが使えるならGPUにデータを送る
                imges = imges.to(device)
                anno_class_imges = anno_class_imges.to(device)
                #print("今のフェイズは",phase)
                
                # multiple minibatchでのパラメータの更新
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imges)
                    loss = criterion(outputs,anno_class_imges)
                    # loss = criterion(
                    #     outputs, anno_class_imges.long()) / batch_multiplier

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()  # 勾配の計算
                        count -= 1  # multiple minibatch

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            t_iter_start = time.time()
                        epoch_train_loss.append(loss.item()) #* batch_multiplier
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss.append(loss.item()) # * batch_multiplier
                    pbar.set_description(f"Epoch: {epoch+1} , loss: {loss.item()}") #sec."{duration},' IoU: {score}")

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        # print('-------------')
        # print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
        #     epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        # print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # 最後のネットワークを保存する
    torch.save(net.state_dict(), './u_net.pth')


net=UNet(3,1)
optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3)
criterion = DiceLoss()
accuracy_metric = IoU()

# 学習・検証を実行する
num_epochs = 30
train_model(net, dataloaders_dict,criterion, optimizer, num_epochs=num_epochs)

"""
#U-"Netモデルの性能評価の確認1
import seaborn as sns
plt.figure(1)
plt.figure(figsize=(15,5))
sns.set_style(style="darkgrid")
plt.subplot(1, 2, 1)
sns.lineplot(x=range(1,num_epochs+1), y=total_train_loss, label="Train Loss")
sns.lineplot(x=range(1,num_epochs+1), y=total_valid_loss, label="Valid Loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("DiceLoss")
 
plt.subplot(1, 2, 2)
sns.lineplot(x=range(1,num_epochs+1), y=total_train_score, label="Train Score")
sns.lineplot(x=range(1,num_epochs+1), y=total_valid_score, label="Valid Score")
plt.title("Score (IoU)")
plt.xlabel("epochs")
plt.ylabel("IoU")
plt.show()

"""