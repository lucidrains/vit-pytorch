from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
# from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


from readDataFromExcel import getDataFromExcelFile
from vit_pytorch.vit_3d import ViT


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def Img_and_Label(data_obj):

    img_list = []
    label_list = []
    file_folder = data_obj.imgRootPath

    data_dict = data_obj.excelData
    for idx in range(len(data_dict)):
        img_list.append(data_dict[idx]["img"])

        cur_label = data_dict[idx]["label"]
        if cur_label == '0':
            label_float = float(0)
        else:
            label_float = float(1)
        label_list.append(label_float)

    uniq_names = []
    num_images = []
    label = []
    label_list_short = []

    for ind, name in enumerate(img_list):
        split_name = name.split("_")
        subj = split_name[0]
        label.append(label_list[ind])

        if subj not in uniq_names:
            uniq_names.append(subj)
            num_images.append(1)
            label_list_short.append(label[-1])
        else:
            index = uniq_names.index(subj)
            num_images[index] += 1

    files = [[]]
    labels = []
    ind = 0
    for idx, subj in enumerate(uniq_names):
        if num_images[uniq_names.index(subj)] != 24:
            print("Subject {} has only {} images".format(subj, num_images[uniq_names.index(subj)]))

        else:
            files.append([])
            for img in range(24):
                if img < 10:
                    img_str = "000" + str(img)
                else:
                    img_str = "00" + str(img)
                files[ind].append(os.path.join(file_folder, (subj + "_" + img_str + ".bmp")).replace("\\", "/"))
            labels.append(label_list_short[ind])
            ind += 1

    files = files[0:-1]
    # files = list(filter(None, files))
    # labels = list(filter(None, labels))
    return files, labels


class MRIDataset(Dataset):
    def __init__(self, data_obj, transform=None):
        files, labels = Img_and_Label(data_obj)
        self.file_list = files
        self.label = labels
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        imgs = self.file_list[idx]
        img = np.zeros((224, 224, 24))
        for idx, cur_img in enumerate(imgs):
            img_here = np.asarray(Image.open(cur_img))
            assert img_here.dtype == 'uint8'
            img[:, :, idx] = img_here / (2**8)

        img = np.float32(img)
        label = np.float32(self.label)

        img_transformed = self.transform(img)
        label = self.label[idx]

        return img_transformed, label


if __name__ == '__main__':

    batch_size = 12
    epochs = 100
    lr = 3e-5
    gamma = 0.7
    seed = 42

    seed_everything(seed)

    device = 'cpu'

    n_folds = 10
    cur_dir = os.getcwd()
    print(f"Current Directory: {cur_dir}")
    os.makedirs(os.path.join(cur_dir, "saved_models"), exist_ok=True)

    excelFilePath = os.path.join(cur_dir,'Fold_Split.xlsx')
    imgRootPath = "C:/Users/jrb187/PycharmProjects/FITNet/subset_data/2D_Images"

    # Transforms to data
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    for fold in range(n_folds):

        excel_sheet_name_train = 'train_fold' + str(fold)
        excel_sheet_name_test = 'valid_fold' + str(fold)

        train_obj = getDataFromExcelFile(excelFilePath=excelFilePath, imgRootPath=imgRootPath, excelSheetName=excel_sheet_name_train)
        test_obj = getDataFromExcelFile(excelFilePath=excelFilePath, imgRootPath=imgRootPath, excelSheetName=excel_sheet_name_test)

        train_dataset = MRIDataset(train_obj, transform=train_transforms)
        test_dataset = MRIDataset(test_obj, transform=val_transforms)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        model = ViT(image_size=224, channels =1, frames=24, image_patch_size=16, frame_patch_size=1, num_classes=2,
                    dim=14*14*24, depth=6, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)

        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for data, label in train_loader:

                # Add 1 (channel)
                data = data.unsqueeze(1)
                assert data.shape == (batch_size, 1, 24, 224, 224)

                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

                torch.cuda.empty_cache()

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in test_loader:
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(test_loader)
                    epoch_val_loss += val_loss / len(test_loader)

            print(
                f"Fold : {fold+1} - Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )

        torch.save(model.state_dict(), './saved_models/{}.pt'.format("fold" + str(fold+1)))

