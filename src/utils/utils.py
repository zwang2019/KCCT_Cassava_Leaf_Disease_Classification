'''
Some util functions
Part of the code is referenced from Kaggle
'''

import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from . import fmix
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import copy


def seed_everything(seed):
    '''All kinds of random seeds are fixed to facilitate ablation experiments.
    Args:
        seed :  int
    '''
    # Fixed random seed in scipy
    random.seed(seed)  # Random seed of fixed random library
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fixed randomness of Python hashes (may not valid)
    np.random.seed(seed)  # Fixed random seed for Numpy
    torch.manual_seed(seed)  # Fixed Torch CPU calculating random seeds
    torch.cuda.manual_seed(seed)  # Fixed CUDA calculating random seeds
    torch.backends.cudnn.deterministic = True  # Whether the calculation of the convolution operator is fixed.The underlying TORCH has different libraries to implement the convolution operator
    torch.backends.cudnn.benchmark = True  # Whether to enable automatic optimization and select the fastest convolution calculation method


def create_result_folder(path):
    '''Create a folder according to the path to store all the trained models obtained from this train
    Args:
        path : str  target path to be created, e.g '../models/new_folder'
    '''
    os.makedirs(path)


def get_sub_training_set(original_df, frac=0.15):
    '''Create subset of training csv input to acquire smaller input to save time for parameter optimization attempting
    :param original_df: pandas.dataframe
    :param frac: frac of subset from original dataframe for each label
    :return: sub_df: pandas.dataframe
    '''
    labels = sorted(original_df['label'].unique())
    sub_df = original_df[original_df['label'] == labels[0]].sample(frac=frac, replace=False)
    for l in labels[1:]:
        sub_df = pd.concat([sub_df, original_df[original_df['label'] == l].sample(frac=frac, replace=False)])
    sub_df = sub_df.sample(frac=1).reset_index(drop=True)
    return sub_df


def get_img(path):
    '''Load the image with OpenCV.
    Due to historical reasons, OpenCV reads images in the BGR format (Old TV setting)
    Args:
        path : str  Image file path e.g '../data/train_img/1.jpg'
    '''
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def rand_bbox(size, lam):
    '''Cutmix bbox interception function
    Args:
        size : tuple Image sizes e.g (256,256)
        lam  : float ratio of image left
    Returns:
        The upper-left and lower-right coordinates of the bbox
        int,int,int,int
    '''
    W = size[0]  # the width of the image
    H = size[1]  # the height of the image
    cut_rat = np.sqrt(1. - lam)  # interception ratio (1 - ratio of image left)
    cut_w = np.int(W * cut_rat)  # The width of the bbox
    cut_h = np.int(H * cut_rat)  # The height of the bbox

    cx = np.random.randint(W)  # Uniformly distributed sampling, the X coordinate of the center point of the intercepted bbox is randomly selected
    cy = np.random.randint(H)  # Uniformly distributed sampling, the Y coordinate of the center point of the intercepted bbox is randomly selected

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # The top-left x-coordinate
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # The top-left y-coordinate
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # The lower-right x-coordinate
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # The lower-right y-coordinate
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    '''data loading class
    Attributes:
        __len__ : lenghth of data samples
        __getitem__ : Index function
    '''
    def __init__(
            self,
            df,
            data_root,
            transforms=None,
            output_label=True,
            label_smoothing=True,
            do_fmix=False,
            fmix_params={
                'alpha': 1.,
                'decay_power': 3.,
                'shape': (512, 512),
                'max_soft': 0.3,
                'reformulate': False
            },
            fmix_probability=0.5,
            do_cutmix=False,
            cutmix_params={
                'alpha': 1,
            },
            cutmix_probability=0.5):
        '''
        Args:
            df : DataFrame , The file name and label of the sample image
            data_root : str , The file path where the image is located, absolute path
            transforms : object , Image augmentation
            output_label : bool , Whether output labels
            label_smoothing : bool , Whether label_smoothing
            do_fmix : bool , Whether to use fmix
            fmix_params :dict , fmix parameters {'alpha':1.,'decay_power':3.,'shape':(256,256),'max_soft':0.3,'reformulate':False}
            do_cutmix : bool, Whether to use cutmix
            cutmix_params : dict , cutmix parameters {'alpha':1.}
        Raises:

        '''
        super().__init__()
        self.df = df.reset_index(drop=True).copy()  # Regenerate index
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.fmix_probablity = fmix_probability
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.cutmix_probability = cutmix_probability
        self.output_label = output_label
        self.label_smoothing = label_smoothing
        if output_label:
            self.labels = self.df['label'].values
            if label_smoothing:

                if not isinstance(self.labels, (list, np.ndarray)):
                    raise ValueError("labels must be 1-D list or array")
                self.labels = torch.LongTensor(self.labels).view(-1, 1)

                zeros_tensor = torch.zeros(self.df.shape[0], self.df['label'].max() + 1)
                filled = zeros_tensor.fill_(0.05 / self.df['label'].max())
                self.labels = filled.scatter_(1, self.labels, 0.95) # Generate the label smoothing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        '''
        Args:
            index : int , index
        Returns:
            img, target(optional)
        '''
        if self.output_label:
            target = self.labels[index]

        img = get_img(
            os.path.join(self.data_root,
                         self.df.loc[index]['image_id']))  # join the path and load the image

        if self.transforms:  # Using image augmentation
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(
                0., 1., size=1)[0] > (1 - self.fmix_probablity):  # 50% chance of triggering FMIX data augmentation (probability can be modified)

            with torch.no_grad():
                lam, mask = fmix.sample_mask(
                    **self.fmix_params)  # Can be modified, which uses the clip to specify the upper and lower limits

                fmix_ix = np.random.choice(self.df.index,
                                           size=1)[0]  # Randomly select the images to mix
                fmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                img = mask_torch * img + (1. - mask_torch) * fmix_img  # Mix the picture

                rate = mask.sum() / float(img.size)  # Get the rate of Mix
                target = rate * target + (
                    1. - rate) * self.labels[fmix_ix]  # Target to mix (should use one-hot first !)

        if self.do_cutmix and np.random.uniform(
                0., 1., size=1)[0] > (1 - self.cutmix_probability):  # 50% chance to trigger cutmix data augmentation (probability can be modified)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(
                    np.random.beta(self.cutmix_params['alpha'],
                                   self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_img.shape[:2], lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2,
                                                        bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) *
                            (bby2 - bby1) / float(img.size))  # Get the rate of Mix
                target = rate * target + (
                    1. - rate) * self.labels[cmix_ix]  # Target to mix (should use one-hot first !)

        if self.output_label:
            return img, target
        else:
            return img


def prepare_dataloader(df, trn_idx, val_idx, data_root, trn_transform,
                       val_transform, bs, n_job):
    '''Multithreaded data generator
    Args:
        df : DataFrame , The file name and label of the sample image
        trn_idx : ndarray , Training set index list
        val_idx : ndarray , Validation set index list
        data_root : str , The path of the image file
        trn_transform : object , Training set image augmentation
        val_transform : object , Validation set image augmentation
        bs : int , Number of batchsizes per time !!!
        n_job : int , Number of threads in use
    Returns:
        train_loader, val_loader , Data generators for training sets and validation sets
    '''
    train_ = df.loc[trn_idx, :].reset_index(drop=True)  # Regenerate index
    valid_ = df.loc[val_idx, :].reset_index(drop=True)  # Regenerate index

    train_ds = CassavaDataset(train_,
                              data_root,
                              transforms=trn_transform,
                              output_label=True,
                              label_smoothing=True,
                              do_fmix=False,
                              do_cutmix=False)
    valid_ds = CassavaDataset(valid_,
                              data_root,
                              transforms=val_transform,
                              output_label=True,
                              label_smoothing=True,
                              do_fmix=False,
                              do_cutmix=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=n_job,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=n_job,
    )

    return train_loader, val_loader


def train_one_epoch(epoch,
                    model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    device,
                    scaler,
                    scheduler=None,
                    schd_batch_update=False,
                    accum_iter=2):
    '''The training function for each epoch
    Args:
        epoch : int , which epoch now
        model : object, the imported architecture of model
        loss_fn : object, loss function
        optimizer : object, optimization method
        train_loader : object, Training set data generator
        scaler : object, Gradient amplifier
        device : str , Training devices e.g 'cuda:0'
        scheduler : object , Learning rate adjustment strategy
        schd_batch_update : bool, If true, adjust each batch, otherwise wait until the epoch has finished
        accum_iter : int , Gradient accumulation
    '''

    model.train()  # Training Mode

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar

    for step, (imgs, image_labels) in pbar:  # Iterate through each batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device)
        #TODO
        # test double()
        # image_labels = image_labels.to(device).double()
        with autocast():  # Enable automatic mix accuracy
            image_preds = model(imgs)  # Propagate forward and calculate the predicted value
            loss = loss_fn(image_preds, image_labels)  # Calculating the loss

        scaler.scale(loss).backward()  # scale gradient
        # loss regularization with exponential average
        if running_loss is None:
            running_loss = copy.copy(loss)
        else:
            running_loss = running_loss * .99 + copy.copy(loss) * .01

        if ((step + 1) % accum_iter == 0) or ((step + 1) == len(train_loader)):
            scaler.step(
                optimizer)  # Unscale gradient, if the gradient does not overflow, use opt to update the gradient, otherwise do not update
            scaler.update()  # Wait for the next gradient scale
            optimizer.zero_grad()  # Empty Gradient

            if scheduler is not None and schd_batch_update:  # Learning rate adjustment strategies
                scheduler.step()

        # print loss
        description = f'epoch {epoch} loss: {running_loss:.4f}'
        pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:  # Learning rate adjustment strategies
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    '''Validation set inference
    Args:
        epoch : int, which epoch now
        model : object, the imported architecture of model
        loss_fn : object, loss function
        val_loader ： object, Validation set data generator
        device : str , Validating devices e.g 'cuda:0'
    '''

    model.eval()  # inference mode

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))  # progress bar

    for step, (imgs, image_labels) in pbar:  # Iterate through each batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # Propagate forward and calculate the predicted value
        image_preds_all += [
            torch.argmax(image_preds, 1).detach().cpu().numpy()
        ]  # Get the prediction labels
        image_targets_all += [torch.argmax(image_labels, 1).detach().cpu().numpy()]  # Get the true labels

        loss = loss_fn(image_preds, image_labels)  # Calculating loss

        loss_sum += copy.copy(loss) * image_labels.shape[0]  # Calculating loss sum
        sample_num += image_labels.shape[0]  # sample size

        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'  # Print Average Loss
        pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format(
        (image_preds_all == image_targets_all).mean()))  # Printing accuracy


if __name__ == '__main__':
    pass
