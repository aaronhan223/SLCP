import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import gc
gc.collect()
from torchvision.models import *
from torchvision import transforms
import pretrainedmodels
import config
import numpy as np
import pandas as pd
import torch.optim as optim
from datetime import datetime, timedelta
from fastai.vision import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pdb
from utils import ImageLoader
cuda = torch.cuda.is_available()
from sklearn.model_selection import train_test_split


def datenum_to_datetime(datenum):
    
    try:
        days = datenum % 1
        hours = days % 1 * 24
        minutes = hours % 1 * 60
        seconds = minutes % 1 * 60
        exact_date = datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)
    
        return exact_date.year
    
    except(ValueError, TypeError, OverflowError):
        
        return np.nan  


class AgeModel(nn.Module):
    def __init__(self, embedding=False):
        super().__init__()
        layers = list(resnet(pretrained=True).children())[:-2]
        layers += [AdaptiveConcatPool2d(), Flatten()]
        #layers += [nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.60)]
        #layers += [nn.Linear(4096, 1024, bias=True), nn.ReLU(inplace=True)]
        #layers += [nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.60)]
        #layers += [nn.Linear(2048, 1024, bias=True), nn.ReLU(inplace=True)]
        #layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.75)]
        #layers += [nn.Linear(1024, 256, bias=True), nn.ReLU(inplace=True)]
        #layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        #layers += [nn.Dropout(p=0.50)]
        #layers += [nn.Linear(512,256 , bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(1024, 512, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(512, 16, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        if not embedding:
            self.agemodel = nn.Sequential(*layers)
        else:
            self.agemodel = nn.Sequential(*layers[-2])
    def forward(self, x):
        return self.agemodel(x).squeeze(-1)


def resnet(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.resnet34(pretrained=pretrained)
    return model


def evaluate(data_loader, model):
    model.eval()
    all_data = next(iter(data_loader))
    inputs = torch.Tensor(all_data['image'].float())
    results = torch.Tensor(all_data['label'].float())
    if cuda:
        inputs, results = inputs.cuda(), results.cuda()
    predictions = torch.squeeze(model(inputs))
    loss = F.mse_loss(predictions, results)
    return loss.data, predictions


def train_epoch(train_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for batch_data in train_loader:
        inputs = torch.Tensor(batch_data['image'].float())
        results = torch.Tensor(batch_data['label'].float())
        if cuda:
            inputs, results = inputs.cuda(), results.cuda()
        optimizer.zero_grad()
        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.mse_loss(predictions, results)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    return total_loss


def main():
    seed = 42
    np.random.seed(seed)
    batch_size = 32
    num_epoch = 200

    img_list = pd.read_csv('./datasets/img_labels.csv', index_col=0)
    train, test = train_test_split(np.arange(img_list.shape[0]), test_size=config.DataParams.test_ratio, random_state=42)
    train_idx, valid_idx, test_idx = train[:int(0.5 * len(train))], train[int(0.5 * len(train)):], test
    img_list_train, img_list_valid, img_list_test = img_list.iloc[train_idx], img_list.iloc[valid_idx], img_list.iloc[test_idx]
    img_list_train.to_csv('./datasets/img_labels_train.csv')
    img_list_valid.to_csv('./datasets/img_labels_valid.csv')
    img_list_test.to_csv('./datasets/img_labels_test.csv')

    ImageDataset_train = ImageLoader(annotations_file='./datasets/img_labels_train.csv',
                                img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                transform=torch.nn.Sequential(
                                                    transforms.RandomRotation(degrees=10),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomApply(
                                                        torch.nn.ModuleList([
                                                            transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                            ]), p=0.7),
                                                    transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge'),
                                                    transforms.Resize(128),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    )
                                                )
    ImageDataset_valid = ImageLoader(annotations_file='./datasets/img_labels_valid.csv',
                                    img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                    transform=torch.nn.Sequential(
                                                        transforms.RandomRotation(degrees=10),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomApply(
                                                            torch.nn.ModuleList([
                                                                transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                                ]), p=0.7),
                                                        transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge'),
                                                        transforms.Resize(128),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                        )
                                                    )
    ImageDataset_test = ImageLoader(annotations_file='./datasets/img_labels_test.csv',
                                    img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                    transform=torch.nn.Sequential(
                                                        transforms.RandomRotation(degrees=10),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomApply(
                                                            torch.nn.ModuleList([
                                                                transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                                ]), p=0.7),
                                                        transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge'),
                                                        transforms.Resize(128),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                        )
                                                    )    
    train_loader = DataLoader(ImageDataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = DataLoader(ImageDataset_valid, batch_size=len(ImageDataset_valid), num_workers=0, shuffle=True)
    test_loader = DataLoader(ImageDataset_test, batch_size=len(ImageDataset_test), num_workers=0, shuffle=True)
    model = AgeModel()
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-5)
    best_loss = 1e10
    for i in range(1, num_epoch + 1):
        train_loss = train_epoch(train_loader, model, optimizer)
        valid_loss, _ = evaluate(valid_loader, model)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 
                        f'./saved_models/best_model.pt')
        if i % 10 == 0:
            torch.save({'epoch': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 
                        f'./saved_models/model_epoch_{i}.pt')
            print('Epoch [%d]: Train loss: %.3f.' % (i, train_loss))


if __name__ == '__main__':
    main()