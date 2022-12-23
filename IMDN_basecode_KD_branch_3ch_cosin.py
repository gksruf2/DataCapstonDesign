import sys
import os
import natsort
import time
import datetime
import random
import argparse
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import cv2

from torch import rand, manual_seed
from PIL import Image #, ImageFile
# from IPython import display
from torch.optim import SGD, Adam, lr_scheduler
from torch import nn, FloatTensor, set_grad_enabled, save, uint8, cuda, device, no_grad, stack, mean
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, io, transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#from network_swinir import SwinIR
from utils.CustomUtils import *

#ImageFile.LOAD_TRUNCATED_IMAGES = True

device = device('cuda:0' if cuda.is_available() else 'cpu')
print(device)

class FLIR_Data(Dataset):
    def __init__(self, LR_path, HR_path, transform, flip=False, grayscale=False):
        self.LR_data = LR_path
        self.HR_data = HR_path

        self.len = len(LR_path)
        self.transform = transform
        self.horizontalflip = transforms.RandomHorizontalFlip(p = 1)
        self.verticalflip = transforms.RandomVerticalFlip(p = 1)
        self.flip = flip    # True / False
        self.grayscale = grayscale  # True / False

    def __getitem__(self, index):
        if self.grayscale:
            image_LR = Image.fromarray(np.load(self.LR_data[index]))
            image_HR = Image.fromarray(np.load(self.HR_data[index]))
            image_LR = transforms.functional.to_grayscale(image_LR, num_output_channels=3)
            image_HR = transforms.functional.to_grayscale(image_HR, num_output_channels=1)
        else:
            image_LR = Image.fromarray(np.load(self.LR_data[index]))
            image_HR = Image.fromarray(np.load(self.HR_data[index]))

        if self.flip:
            if rand(1) > 0.5:
                image_LR, image_HR = self.horizontalflip(image_LR), self.horizontalflip(image_HR)
            if rand(1) > 0.5:
                image_LR, image_HR = self.verticalflip(image_LR), self.verticalflip(image_HR)
        image_LR, image_HR = self.transform(image_LR), self.transform(image_HR)
        return image_LR, image_HR

    def __len__(self):
        return self.len
    
def fit(model, criterion,optimizer, train_loader, val_loader, num_epochs, saving_name):
    train_size = len(train_loader)
    val_size = len(val_loader)
    start = time.time()
    train_psnr_over_time = []
    val_psnr_over_time = []
    train_ssim_over_time = []
    val_ssim_over_time = []
    best_model = model.state_dict()
    best_psnr = 0
    best_ssim = 0
    ori_norm = None
    aft_norm = None
    bef_ = None
    last_psnr = 0
    last_ssim = 0

    Train_Verbose = False

    # each epoch has a training and validation phase
    for epoch in range(1, num_epochs+1):
        for phase in ['train','val']:
            if phase == 'train':
                data_loader = train_loader
                model.train()                    # set the model to train mode
                end = time.time()
            else:
                data_loader = val_loader
                model.eval()                    # set the model to evaluate mode
                end = time.time()
            
            sum_of_epoch_psnr = 0.0
            sum_of_epoch_ssim = 0.0
            loss_batch = []
            # iterate over the data
            for step_,(LR,HR) in enumerate(data_loader):
                LR = LR.to(device)
                HR = HR.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                batch_psnr = 0 # psnr of each batch
                batch_ssim = 0
                # forward
                with set_grad_enabled(phase == 'train'):
                    #LR_BI = nn.functional.interpolate(LR, size=(480, 640), mode='bicubic')
                    #LR_BI = LR_BI.to(device)
                    if phase == 'train':    # LR_channel : 3, HR_channel : 3
                        outputs = model(LR)     # original(3) to 3
                    else:   # LR_channel : 3, HR_channel : 1
                        outputs = model(LR)     # stack(1x3) to 3
                        outputs = mean(outputs, dim=1, keepdim=True)    # 3 to 1(mean)
                    """
                    LR_3 = stack([LR[:,0,:,:],LR[:,0,:,:],LR[:,0,:,:]],dim=1)
                    outputs = model(LR_3)
                    outputs = nn.functional.interpolate(outputs, size=(480, 640), mode='bicubic')
                    outputs = mean(outputs, dim=1, keepdim=True)"""
                    #outputs = outputs.to(device)

                    loss = criterion(outputs, HR)
                    loss_batch.append(loss.item())

                    if phase == 'train':
                      loss.backward()
                      optimizer.step()

                    last_psnr = 0
                    last_ssim = 0

                    if phase == 'train' and Train_Verbose:
                        original = HR[0].detach().cpu().numpy()
                        trained = outputs[0].detach().cpu().numpy()

                        ori_norm = normalize_ndarray_256(original)
                        tra_norm = normalize_ndarray_256(trained)

                        ori_norm = ori_norm.astype(np.uint8).transpose((1, 2, 0))
                        aft_norm = tra_norm.astype(np.uint8).transpose((1, 2, 0))

                        last_psnr = peak_signal_noise_ratio(ori_norm, aft_norm)  # psnr of single image
                        #last_ssim, _ = structural_similarity(ori_norm, aft_norm, full=True, multichannel=True)
                        last_ssim, _ = structural_similarity(ori_norm, aft_norm, full=True, channel_axis=-1)
                    elif phase == 'val':
                        original = HR.detach().cpu().numpy()
                        trained = outputs.detach().cpu().numpy()

                        ori_norm = normalize_ndarray_256(original)
                        tra_norm = normalize_ndarray_256(trained)

                        ori_norm = ori_norm.astype(np.uint8).transpose((0, 2,3,1))
                        aft_norm = tra_norm.astype(np.uint8).transpose((0, 2,3,1))

                        for i in range(HR.shape[0]):
                            last_psnr += peak_signal_noise_ratio(ori_norm[i], aft_norm[i])  # psnr of single image
                            #last_ssim += structural_similarity(ori_norm[i], aft_norm[i], full=True, multichannel=True)[0]
                            last_ssim += structural_similarity(ori_norm[i], aft_norm[i], full=True, channel_axis=-1)[0]
                        #ori_norm = ori_norm[0]
                        #bef_ = LR[0]
                        #aft_norm = aft_norm[0]

                    # backward + optimize only if in training phase
                    
                    sum_of_epoch_psnr += last_psnr
                    sum_of_epoch_ssim += last_ssim
                    
                    if step_ % 100 == 0:
                      if phase == 'train' and Train_Verbose:
                        print(f'batch_psnr: {last_psnr:.3f}, batch_ssim: {last_ssim:.3f}')
                        print(f'epoch_psnr: {sum_of_epoch_psnr / (step_ + 1):.3f}, epoch_ssim: {sum_of_epoch_ssim / (step_ + 1):.3f}')
                        print(f"Now phase is {phase}, {step_}/{train_size + val_size}")
                      elif phase == 'val':
                        print(f'batch_psnr: {last_psnr/batch_size:.3f}, batch_ssim: {last_ssim/batch_size:.3f}')
                        print(f'epoch_psnr: {sum_of_epoch_psnr / (batch_size* (step_ + 1)):.3f}, epoch_ssim: {sum_of_epoch_ssim / (batch_size*(step_ + 1)):.3f}')
                        print(f"Now phase is {phase}, {step_ + train_size}/{train_size + val_size}")

            if phase == 'train' and Train_Verbose:
                epoch_psnr = sum_of_epoch_psnr / len(train_loader)
                train_psnr_over_time.append(epoch_psnr)
                epoch_ssim = sum_of_epoch_ssim / len(train_loader)
                train_ssim_over_time.append(epoch_ssim)
                print(f'{phase} psnr: {epoch_psnr:.3f}, ssim: {epoch_ssim:.3f}')
                elapsed_time = time.time() - start
                print('==> {:.2f} total elapsed time : \n'.format(elapsed_time))

            elif phase == 'val':
                epoch_psnr = sum_of_epoch_psnr / len(val_data)
                val_psnr_over_time.append(epoch_psnr)
                epoch_ssim = sum_of_epoch_ssim / len(val_data)
                val_ssim_over_time.append(epoch_ssim)
                print(f'{phase} psnr: {epoch_psnr:.3f}, ssim: {epoch_ssim:.3f}')
                elapsed_time = time.time() - start
                print('==> {:.2f} total elapsed time : \n'.format(elapsed_time))
            print(f'loss: {np.mean(loss_batch)}')

            if phase == 'val' and epoch_psnr > best_psnr:
                best_psnr = epoch_psnr
                save(model.state_dict(), './IMDN/model_best_psnr.pt')
                
            if phase == 'val' and epoch_ssim > best_ssim:
                best_ssim = epoch_ssim
                save(model.state_dict(), './IMDN/model_best_ssim.pt')

            save(model.state_dict(), '_model_latest.pt')
        scheduler.step()

        if epoch in [100, 200, 300]:
            evaluate_Student(model, path = './IMDN/save', saving_name=saving_name+str(epoch))

        print('-'*30, 'epoch', epoch, 'train & validation ended', '-' * 30)
    if Train_Verbose:
        psnr = {'train':train_psnr_over_time, 'val':val_psnr_over_time}
        ssim = {'train':train_ssim_over_time, 'val':val_ssim_over_time}
    else:
        psnr = {'train':None, 'val':val_psnr_over_time}
        ssim = {'train':None, 'val':val_ssim_over_time}
    return model, psnr, ssim

def evaluate_Student(model, path, saving_name):
    print(f'evaluate_Student start')
    model.eval()  # setting the model to evaluate mode
    test_psnr = []
    test_ssim = []
    model.load_state_dict(torch.load('./IMDN/model_best_psnr.pt'))
    save(model.state_dict(), path+'/model_best_psnr'+saving_name+'.pt')

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # predicting
        with torch.no_grad():
            outputs = model(inputs)
            outputs = mean(outputs, dim=1, keepdim=True)
            #outputs = model(inputs)

            original = labels.detach().cpu().numpy()
            trained = outputs.detach().cpu().numpy()

            ori_norm = normalize_ndarray_256(original)
            tra_norm = normalize_ndarray_256(trained)

            ori_norm = ori_norm.astype(np.uint8).transpose((0, 2, 3, 1))
            aft_norm = tra_norm.astype(np.uint8).transpose((0, 2, 3, 1))

            for i in range(inputs.shape[0]):
                test_psnr.append(peak_signal_noise_ratio(ori_norm[i], aft_norm[i]))  # psnr of single image
                test_ssim.append(structural_similarity(ori_norm[i], aft_norm[i], full=True, channel_axis=-1)[0])
                
    print("--for best psnr model--")
    print(f'best PSNR: {np.mean(test_psnr):.3f}')
    print(f'best SSIM: {np.mean(test_ssim):.3f}')

    Id = list(range(0, len(test_psnr)))

    prediction = {
        'Id': Id,
        'PSNR': test_psnr,
        'SSIM': test_ssim
    }

    prediction_df = pd.DataFrame(prediction, columns=['Id', 'PSNR', 'SSIM'])
    prediction_df.to_csv(path+'IMDN_psnr'+saving_name+'.csv', index=False)

    test_psnr = []
    test_ssim = []
    model.load_state_dict(torch.load('./IMDN/model_best_ssim.pt'))
    save(model.state_dict(), path+'/model_best_ssim'+saving_name+'.pt')

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # predicting
        with torch.no_grad():
            outputs = model(inputs)
            outputs = mean(outputs, dim=1, keepdim=True)

            original = labels.detach().cpu().numpy()
            trained = outputs.detach().cpu().numpy()

            ori_norm = normalize_ndarray_256(original)
            tra_norm = normalize_ndarray_256(trained)

            ori_norm = ori_norm.astype(np.uint8).transpose((0, 2, 3, 1))
            aft_norm = tra_norm.astype(np.uint8).transpose((0, 2, 3, 1))

            for i in range(inputs.shape[0]):
                test_psnr.append(peak_signal_noise_ratio(ori_norm[i], aft_norm[i]))  # psnr of single image
                test_ssim.append(structural_similarity(ori_norm[i], aft_norm[i], full=True, channel_axis=-1)[0])
                
    print("--for best ssim model--")
    print(f'best PSNR: {np.mean(test_psnr):.3f}')
    print(f'best SSIM: {np.mean(test_ssim):.3f}')

    Id = list(range(0, len(test_psnr)))

    prediction = {
        'Id': Id,
        'PSNR': test_psnr,
        'SSIM': test_ssim
    }
    prediction_df = pd.DataFrame(prediction, columns=['Id', 'PSNR', 'SSIM'])
    prediction_df.to_csv(path+'IMDN_ssim'+saving_name+'.csv', index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Knowledge Distillation Parser')
    parser.add_argument('--optimizer', required=True, type = str, help = 'optimizer function : ADAM or SGD')
    parser.add_argument('--epochs', required=True, type = int, help = 'number of epochs')
    parser.add_argument('--batch', required=True, type = int, help = 'learning rate')
    #parser.add_argument('--alpha', required=True, type = float, help = 'DS_weight between 0 ~ 1')
    parser.add_argument('--KDloss', required=True, type = str, help = 'KD loss function : L1 or L2')
    parser.add_argument('--lr', required=True, type = float, help = 'learning rate')
    parser.add_argument('--name', required=False, type = str, help = 'saving name')
    
    parameters = parser.parse_args()
    
    print('############ KD basecode ##############')
    print(f'current time : {datetime.datetime.now()}')
    print('parameter setting...')
    print(f'optimizer = {parameters.optimizer}')
    print(f'epochs = {parameters.epochs}')
    print(f'batch size = {parameters.batch}')
    #print(f'alpha(DS_weight) = {parameters.alpha}')
    print(f'KD loss function = {parameters.KDloss}')
    print(f'learning rate = {parameters.lr}')
    print(f'saving name = {parameters.name}')
    print(f'######################################')
    
    #################### unpack parser & parameter setting #########################
    
    epochs = parameters.epochs
    batch_size = parameters.batch
    #alpha = parameters.alpha # DS_weight
    lr = parameters.lr
    
    if parameters.KDloss == 'L1':
        loss_function = nn.L1Loss()
    elif parameters.KDloss == 'L2':
        loss_function = nn.MSELoss()
    else:
        raise Exception('invalid loss function hyper-parameter')
    
    input_size = 1
    output_size = 1
    upscale = 4
    
    height = 120
    width = 160
    
    manual_seed(10)

    train_size = 8000
    val_size = 1000
    test_size = 2000

    train_dir_LR = './dataset/numpy_3chan/train/low'
    train_dir_HR = './dataset/numpy_3chan/train/high'
    
    val_dir_LR = './dataset/numpy_3chan/test/low'
    val_dir_HR = './dataset/numpy_3chan/test/high'
    
    test_dir_LR = './dataset/numpy_3chan/valid/low'
    test_dir_HR = './dataset/numpy_3chan/valid/high'
    
    manual_seed(time.time())
    import IMDN_architecture
    model = IMDN_architecture.IMDN(in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4)
    model.to(device)

    """teacher_model = SwinIR(upscale=upscale, in_chans=3, img_size=64, window_size=8,
                           img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                           mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    
    teacher_model.to(device)

    model_path = './pretrained/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
    teacher_model.load_state_dict(torch.load(model_path)['params'], strict = False)
    """
    if parameters.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr, momentum=0.9, weight_decay = 0.0005)
    elif parameters.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr, betas=(0.9, 0.99), eps=1e-08)
    else:
        raise Exception('invalid optimizer hyper-parameter')
        
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[999], gamma=0.5)
    from CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=0.001,  T_up=5, gamma=0.5)

    ############################ parameter setting end ################################

    train_transform = transforms.Compose([transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_name_LR = make_file_list(train_dir_LR, train_size)
    train_name_HR = make_file_list(train_dir_HR, train_size)
    
    val_name_LR = make_file_list(val_dir_LR, val_size)
    val_name_HR = make_file_list(val_dir_HR, val_size)
    
    test_name_LR = make_file_list(test_dir_LR, test_size)
    test_name_HR = make_file_list(test_dir_HR, test_size)

    train_data = FLIR_Data(train_name_LR, train_name_HR, train_transform, flip=True)
    val_data = FLIR_Data(val_name_LR, val_name_HR, val_transform, grayscale=True)
    test_data = FLIR_Data(test_name_LR, test_name_HR, test_transform, grayscale=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f'student model total parameter : {total_params(model)}')
    #print(f'teacher model total parameter : {total_params(teacher_model)}')
    
    model.to(device)
    torch.cuda.empty_cache()
    
    #result_path = f'result/{parameters.optimizer}_{str(epochs)}_{str(alpha)}_{parameters.KDloss}_{str(lr)}'
    
    """result_model, result_psnr, result_ssim = \
        fit_kd(model, teacher_model, optimizer, train_loader, val_loader, 
               epochs = epochs, alpha = alpha, loss_function = loss_function, result_path = result_path)
    """
    criterion = loss_function
    result_model, result_psnr, result_ssim = \
        fit(model, criterion, optimizer, train_loader, val_loader, 
               num_epochs = epochs, saving_name=parameters.name)
    
    print('result_psnr =', result_psnr)
    print('result_ssim =', result_ssim)

    train_psnr = result_psnr['train']
    val_psnr = result_psnr['val']
    train_ssim = result_ssim['train']
    val_ssim = result_ssim['val']
    
    # testing the model
    #predictions = evaluate_Student(model, path = './')
    
    print('All Done!!!\n\n')