
import sys
sys.path.append('../')

from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from torch import nn
from tqdm import tqdm
import torch
import timm
import cv2
import pandas as pd
import numpy as np
from utils import utils
from imp import reload
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
reload(utils)
rand_seed = 666
utils.seed_everything(rand_seed)

###################
train_img_path = r'../data/train_images'
train_csv_path = r'../data/train.csv' 

###################


# Training set augmentation
def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)

# Validation set augmentation
def get_valid_transforms():
    return Compose([
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Resize(CFG['img_size'], CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)



####################

# model constructing
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
    def forward(self, x):
        x = self.model(x)
        return x

#######################



# configuration
CFG = {
    'img_size' : 512,
    'epochs': 15,
    'fold_num': 5,
    'device': 'cuda',
    'model_arch': ['tf_efficientnet_b4_ns', 'resnext50d_32x4d', 'resnext101_32x8d', ],
    'train_bs' : 32,
    'valid_bs' : 32,
    'num_workers' : 4,
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'T_0': 10,
    'min_lr': 1e-6,
}
train = pd.read_csv(train_csv_path)
folds = StratifiedKFold(n_splits=CFG['fold_num'],
                        shuffle=True,
                        random_state=rand_seed).split(
                            np.arange(train.shape[0]), train.label.values)
trn_transform = get_train_transforms()
val_transform = get_valid_transforms()


###########################
first_fold = 0
for fold, (trn_idx, val_idx) in enumerate(folds):
    if fold == 0:

        print('Training with {} started'.format(fold))
        print('Train : {}, Val : {}'.format(len(trn_idx), len(val_idx)))
        train_loader, val_loader = utils.prepare_dataloader(train,
                                                          trn_idx,
                                                          val_idx,
                                                          data_root = train_img_path,
                                                          trn_transform = trn_transform,
                                                          val_transform = val_transform,
                                                          bs = CFG['train_bs'],
                                                          n_job = CFG['num_workers'])

        device = torch.device(CFG['device'])

        model = CassvaImgClassifier(CFG['model_arch'][1],
                                    train.label.nunique(),
                                    pretrained=True).to(device)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=CFG['lr'],
                                     weight_decay=CFG['weight_decay'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=CFG['T_0'],
            T_mult=1,
            eta_min=CFG['min_lr'],
            last_epoch=-1)

        loss_tr = nn.CrossEntropyLoss().to(
            device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(CFG['epochs']):
            utils.train_one_epoch(epoch,
                                model,
                                loss_tr,
                                optimizer,
                                train_loader,
                                device,
                                scaler,
                                scheduler=scheduler,
                                schd_batch_update=False)

            with torch.no_grad():
                utils.valid_one_epoch(epoch,
                                    model,
                                    loss_fn,
                                    val_loader,
                                    device)

            torch.save(
                model.state_dict(),
                '../model/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()

























