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
import sys
import scipy.io
import numpy as np
import pandas as pd
import torch.optim as optim
from datetime import datetime, timedelta
from fastai.vision import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random
import shutil
import pathlib
import pdb
from utils import ImageLoader
cuda = torch.cuda.is_available()


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
    def __init__(self):
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
        self.agemodel = nn.Sequential(*layers)
    def forward(self, x):
        return self.agemodel(x).squeeze(-1)


# def imdb_wiki_preprocess(name, tfms):
#     '''
#     name (str): input data name, can be wiki or imdb
#     '''
#     path = Path('/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/' + name + '_crop/')
#     mat = scipy.io.loadmat('/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/' + name + '_crop/' + name + '.mat')
#     columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", 
#                "face_score", "second_face_score", 'celeb_names', 'celeb_id']
#     instances = mat[name][0][0][0].shape[1]
#     df = pd.DataFrame(index = range(0, instances), columns = columns)

#     for i in mat:
#         if i == name:
#             current_array = mat[i][0][0]
#             for j in range(len(current_array)):
#                 df[columns[j]] = pd.DataFrame(current_array[j][0])

#     df['date_of_birth'] = df['dob'].apply(datenum_to_datetime) 
#     df['age'] = df['photo_taken'] - df['date_of_birth']

#     #remove pictures does not include face
#     df = df[df['face_score'] != -np.inf]

#     #some pictures include more than one face, remove them
#     df = df[df['second_face_score'].isna()]

#     #check threshold
#     df = df[df['face_score'] >= 3.5]

#     df = df.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])

#     #some guys seem to be greater than 100. some of these are paintings. remove these old guys
#     df = df[df['age'] <= 100]

#     #some guys seem to be unborn in the data set
#     df = df[df['age'] > 0]

#     df['age'] = df['age'].apply(lambda x: int(x))
#     df = df.drop(columns=['dob', 'photo_taken'])
#     df_age = df.drop(columns=['gender', 'celeb_names', 'celeb_id'])

#     df_age['full_path'] = df_age['full_path'].str.get(0)
#     df_age.dropna(axis=0, inplace=True)
#     df_age['age'] = df_age['age'].apply(lambda x: int(x))
#     if name == 'imdb':
#         data = ImageList.from_df(df_age, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(
#                                 label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                                 size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#         return data, df_age
#     else:
#         data_small = ImageList.from_df(df_age, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(
#                      label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                      size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#         data_big = ImageList.from_df(df_age, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(
#                    label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                    size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#         return data_small, data_big, df_age


# def extract_age(filename):
#     return float(filename.stem.split('_')[0])


# def utk_preprocess(tfms):
#     path_utk = Path('/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/UTKFace')
#     data_utk_small = ImageList.from_folder(path_utk).split_by_rand_pct(0.2, seed=42).label_from_func(
#                     extract_age, label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                     size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#     data_utk_big = ImageList.from_folder(path_utk).split_by_rand_pct(0.2, seed=42).label_from_func(
#                     extract_age, label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                     size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#     return data_utk_small, data_utk_big


# def appa_preprocess(tfms):
#     df_appa = pd.read_csv('/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/appa-real-face-cropped/labels.csv')
#     df_appa.rename(columns = {"file_name":"full_path", "real_age":"age"}, inplace=True)
#     df_appa['age'] = df_appa['age'].apply(lambda x: int(float(x)))
#     path = Path('/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/appa-real-face-cropped/final_files/final_files/')

#     data_appa_small = ImageList.from_df(df_appa, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(
#                                         label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                                         size=128).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#     data_appa_big = ImageList.from_df(df_appa, path, cols=['full_path'], folder ='.').split_by_rand_pct(0.2, seed=42).label_from_df(
#                                         label_cls=FloatList).transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', 
#                                         size=256).databunch(bs=64*2,num_workers=0).normalize(imagenet_stats)
#     return data_appa_small, data_appa_big, df_appa


def resnet(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.resnet34(pretrained=pretrained)
    return model


# class L1LossFlat(nn.SmoothL1Loss):
#     def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
#         return super().forward(input.view(-1), target.view(-1))


def train(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    train_features, train_labels = next(iter(data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    # for data, label in data_loader:
    #     inputs = torch.Tensor(data.float())
    #     results = torch.Tensor(label.float())
    #     if cuda:
    #         inputs, results = inputs.cuda(), results.cuda()
    #     optimizer.zero_grad()
    #     predictions = model(inputs)
    #     predictions = torch.squeeze(predictions)
    #     loss = F.mse_loss(predictions, results)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += loss.data
    # return total_loss


def main():
    seed = 42
    np.random.seed(seed)

    # tfms = get_transforms(max_rotate=10., max_zoom=1., max_lighting=0.20, do_flip=False,
    #                       max_warp=0., xtra_tfms=[flip_lr(), brightness(change=(0.3, 0.60), p=0.7), contrast(scale=(0.5, 2), p=0.7),
    #                                               crop_pad(size=600, padding_mode='border', row_pct=0.,col_pct=0.),
    #                                               rand_zoom(scale=(1.,1.5)), rand_crop(), perspective_warp(magnitude=(-0.1,0.1)),
    #                                               symmetric_warp(magnitude=(-0.1,0.1))])
    # data_wiki_small, data_wiki_big, df_age_wiki = imdb_wiki_preprocess('wiki', tfms)
    # data_imdb, df_age_imbd = imdb_wiki_preprocess('imdb', tfms)
    # data_utk_small, data_utk_big = utk_preprocess(tfms)
    # data_appa_small, data_appa_big, df_appa = appa_preprocess(tfms)
    
    # opt_func = partial(optim.Adam, betas=(0.9,0.99), eps=1e-5)
    # df_utk_small = data_utk_small.to_df()
    # df_utk_small.rename(columns = {"x":"full_path", "y":"age"}, inplace=True)
    # df_utk_small['age'] = df_utk_small['age'].apply(lambda x: int(float(x)))

    # src_wiki = '/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/wiki_crop/'
    # src_utk = '/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/UTKFace/'
    # src_appa = '/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/appa-real-face-cropped/final_files/'
    # dest = '/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk/'

    # pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    # os.listdir(dest)
    # for root, dirs, files in os.walk(src_wiki):
    #     for file in files:
    #         path_file = os.path.join(root,file)
    #         shutil.copy2(path_file, dest)

    # for root, dirs, files in os.walk(src_utk):
    #     for file in files:
    #         path_file = os.path.join(root,file)
    #         shutil.copy2(path_file,dest)
    
    # for root, dirs, files in os.walk(src_appa):
    #     for file in files:
    #         path_file = os.path.join(root,file)
    #         shutil.copy2(path_file,dest)

    # df_age_wiki['full_path'] = df_age_wiki['full_path'].str[3:]
    # frames = [df_age_wiki, df_utk_small, df_appa]
    # df_wiki_utk_appa = pd.concat(frames)
    # df_wiki_utk_appa = df_wiki_utk_appa[df_wiki_utk_appa['age'] <= 100]
    # df_wiki_utk_appa = df_wiki_utk_appa[df_wiki_utk_appa['age'] > 0]
    # df_wiki_utk_appa['age'] = df_wiki_utk_appa['age'].astype(int)
    # df_wiki_utk_appa.index = range(df_wiki_utk_appa.shape[0])
    # df_wiki_utk_appa.to_csv(path_or_buf='./datasets/img_labels.csv')
    batch_size = 32
    num_epoch = 50
    ImageDataset = ImageLoader(annotations_file='./datasets/img_labels.csv',
                                img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                transform=torch.nn.Sequential(
                                                    transforms.RandomRotation(degrees=10),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomApply(
                                                        torch.nn.ModuleList([
                                                            transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                            ]), p=0.7),
                                                    transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge')
                                                    )
                                                )
    data_loader = DataLoader(ImageDataset, batch_size=batch_size, num_workers=1, shuffle=False)
    model = AgeModel()
    optimizer = optim.Adam(model.parameters(), lr=2e-2, betas=(0.9, 0.99), eps=1e-5)
    for i in range(1, num_epoch + 1):
        train_loss = train(data_loader, model, optimizer)
        # if i % 10 == 0:
        print('Epoch [%d]: Train loss: %.3f.' % (i, train_loss))

    # TODO: start from df_wiki_utk_appa, write own dataloader and training functions
    # path_wiki_utk_appa = Path('/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk/')
    # data_wiki_small_src = (ImageList.from_df(df_wiki_utk_appa, path_wiki_utk_appa, cols=['full_path'], folder='.')
    #                         .split_by_rand_pct(0.2, seed=42).label_from_df(label_cls=FloatList))
    # data_wiki_small = (data_wiki_small_src.transform(tfms, resize_method=ResizeMethod.CROP, padding_mode='border', size=128)
    #                     .databunch(bs=32,num_workers=0).normalize(imagenet_stats))
    # pdb.set_trace()
    # model = AgeModel()
    # learn = Learner(data_wiki_small, model, model_dir = "/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/imdb_wiki/model", 
    #                 opt_func=opt_func, bn_wd=False, metrics=root_mean_squared_error, callback_fns=[ShowGraph]).mixup(stack_y=False, alpha=0.2)

    # learn.loss_func = L1LossFlat()
    # learn.split([model.agemodel[4],model.agemodel[6],model.agemodel[8]])
    # # learn.freeze_to(-1)
    # # learn.lr_find()
    # # learn.recorder.plot(suggestion = True)
    # lr = 2e-2
    # learn.fit_one_cycle(5, max_lr=slice(lr), wd=(1e-6, 1e-4, 1e-2, 1e-1), pct_start=0.5, callbacks=[SaveModelCallback(learn)])
    # learn.save('first_head_resnet34')
    # learn.load('first_head_resnet34')
    # learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot(suggestion = True)
    # learn.fit_one_cycle(5, max_lr=slice(1e-6, lr/5), wd=(1e-6, 1e-4, 1e-2, 1e-1), 
    #                     callbacks=[SaveModelCallback(learn)], pct_start=0.5)
    # x,y = next(iter(learn.data.train_dl))
    # learn.save('first_body_resnet34')
    # learn.load('first_body_resnet34')
    # learn.show_results()
    # img = open_image('../input/picture-2/55933635_1035600596650844_9173826226136023040_n.jpg')
    # x = learn.predict(img)
    # print('result: ', x)


if __name__ == '__main__':
    main()