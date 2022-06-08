import os
import tarfile
import urllib.request

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.utils.data as data
from PIL import Image
import glob

def get_augmentation(phase):
    if phase == "train":
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            albu.RandomBrightnessContrast()
        ]
        return albu.Compose(train_transform)

    if phase=="valid":
        return None

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

#ここは
def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))


class VOCDataset(data.Dataset):
    CLASSES =  ["background","main_stem","stem"]

    def __init__(self, images_path, masks_path, segment_class, 
                 augmentation=None, preprocessing=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.segment_class = segment_class
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, i):

        # 元画像の読み込み、整形
        image = Image.open(self.images_path[i])
        image = crop_to_square(image)
        image = image.resize((128,128), Image.ANTIALIAS)
        image = np.asarray(image)

        # maskの読み込み、整形
        masks = Image.open(self.masks_path[i])
        masks = crop_to_square(masks)
        masks = masks.resize((128,128), Image.ANTIALIAS)
        masks = np.asarray(masks)

        # maskデータの境界線を表す255は扱いにくいので21に変換
        masks = np.where(masks == 255, 21, masks)

        # maskデータを正解ラベル毎の1hotに変換
        cls_idx = [self.CLASSES.index(cls) for cls in self.segment_class]
        masks = [(masks == idx) for idx in cls_idx]
        mask = np.stack(masks, axis=-1).astype("float")

        # augmentationの実行
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # 前処理の実行
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

# モデルの各種設定
ENCODER = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax2d"
PREDICT_CLASS = ["background","main_stem","stem"]
DEVICE = "cuda"
BATCH_SIZE = 8

# Unet++でモデル作成
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(PREDICT_CLASS), 
    activation=ACTIVATION,
)

# # Unet++でモデル作成
# model = smp.UnetPlusPlus(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=len(PREDICT_CLASS), 
#     activation=ACTIVATION,
# )

# encoderに合わせた前処理の取得
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# データ周りを格納する辞書
data_info = {}
rootpath = r"./data3"

for phase in ["train", "val"]:
    # 画像のpath
    # id_names = rootpath + rf"ImageSets/Segmentation/{phase}.txt"
    data_info[f"{phase}_img_path"] = glob.glob(rootpath+f'/{phase}/*.png')#[rootpath + rf"JPEGImages/{file.strip()}.jpg" for file in open(id_names)]
    data_info[f"{phase}_mask_path"] = glob.glob(rootpath+f'/{phase}_label/*.png')#[rootpath + rf"SegmentationClass/{file.strip()}.png" for file in open(id_names)]
    # print(len(data_info["train_img_path"]))
    # print(len(data_info["train_mask_path"]))
    # Dataset
    data_info[f"{phase}_dataset"] = VOCDataset(
            data_info[f"{phase}_img_path"], 
            data_info[f"{phase}_mask_path"], 
            segment_class=PREDICT_CLASS,
            augmentation=get_augmentation(phase), 
            preprocessing=get_preprocessing(preprocessing_fn)
            )
    
    # DataLoader
    shuffle = True if phase=="train" else False
    data_info[f"{phase}_dataloader"] = data.DataLoader(
        data_info[f"{phase}_dataset"], 
        batch_size=BATCH_SIZE, 
        shuffle=shuffle)

# 学習時の各種設定
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5)]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE
)

patience = 5 # 5Epcoch以上連続でモデル精度が向上しなければEarly Stopping

max_score = 0
for i in range(4000):
    
    print(f"Epoch:{i+1}")
    train_logs = train_epoch.run(data_info["train_dataloader"])
    valid_logs = valid_epoch.run(data_info["val_dataloader"])
    
    # IoUスコアが最高値が更新されればモデルを保存
    if max_score < valid_logs["iou_score"]:
        max_score = valid_logs["iou_score"]
        torch.save(model, "./best_model.pth")
        print("Model saved!")
        early_stop_counter = 0

    else:
        early_stop_counter += 1
        print(f"not improve for {early_stop_counter}Epoch")
        if early_stop_counter==patience:
            print(f"early stop. Max Score {max_score}")
            break

    # 適当なタイミングでlearning rateの変更
    if i == 10:
        optimizer.param_groups[0]["lr"] = 1e-5
        print("Decrease decoder learning rate to 1e-5")